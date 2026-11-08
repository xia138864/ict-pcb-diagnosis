# -*- coding: utf-8 -*-
"""
属性引导的零样本未见故障诊断（Attribute-based Zero-Shot）
=========================================================
思路（让知识图谱真正成为贡献）：

  知识图谱给出「故障 --> 可观测表现」的两跳映射
      fault --has_phenomenon--> phenomenon --manifests_as--> manifestation
  该映射是**跨故障共享**的（如 open / cold_solder / wrong_part 都表现为 voltage_loss），
  因此可以在可见故障上训练「测量向量 --> 属性存在与否」的检测器，
  再按属性组合匹配未见故障，实现零样本识别。

  DAP 打分：s(c) = Σ_j [ a_cj·log p_j(x) + (1 - a_cj)·log(1 - p_j(x)) ]

融合方式：每个属性带一个向量表示，与数据特征经**逐维门控融合**后再判属性是否存在
（复用论文核心模块，保持方法一致性）。

对照：
  · 门控 vs 无门控（拼接）
  · KG 属性矩阵 vs 随机属性矩阵（同密度行置换）
  · KG 属性向量 vs 随机属性向量

用法：
  python zeroshot_attr.py --k-list 20 50 --seeds 2024 2025 2026 2027 2028
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASSES, NUM_CLASSES, load_data, SpiceDataset, AddNoise
from data_encoder import MLPEncoder
from gating import GatingFusion
from train import set_protocol_seed

KG_DIR = os.path.join(_ROOT, 'kg')
OUT_JSON = os.path.join(_ROOT, 'results_zeroshot_attr.json')
D_H = 64
N_ATTR = 10

# 三个留出折：每折 3 个未见类别（随机猜测水平 33.3%）。
# 折设计准则（属性可辨识性）：每个属性在可见类中至少有一个正例，
# 否则该属性的检测器没有训练信号（如 fluctuation 仅由 intermittent 提供，
# freq_low 仅由 drift 提供，故 drift / missing / intermittent 必须保留在可见集）。
FOLDS = {
    'FA': ['open', 'cold_solder', 'bridge'],
    'FB': ['short', 'cold_solder', 'wrong_part'],
    'FC': ['bridge', 'cold_solder', 'wrong_part'],
}

VARIANTS = {
    'Ours-Attr (KG属性矩阵+门控)': dict(gate=True,  attr_emb='kg',     matrix='kg'),
    '属性+无门控':                 dict(gate=False, attr_emb='kg',     matrix='kg'),
    '随机属性矩阵 (门控)':          dict(gate=True,  attr_emb='kg',     matrix='random'),
    '随机属性向量 (门控)':          dict(gate=True,  attr_emb='random', matrix='kg'),
}


def load_attribute_matrix():
    d = json.load(open(os.path.join(KG_DIR, 'class_attributes.json'), encoding='utf-8'))
    attrs = d['attributes']
    M = np.array([[d['class_attributes'][c][a] for a in attrs] for c in CLASSES], dtype=np.float32)
    return attrs, M


def permute_matrix(M, seed):
    """随机属性矩阵对照：行置换（保持密度与向量多重集，打乱「类别→属性」对应）。"""
    rng = np.random.default_rng(seed)
    return M[rng.permutation(M.shape[0])]


class AttrZeroShotModel(nn.Module):
    def __init__(self, encoder, n_attr=N_ATTR, d_h=D_H, use_gate=True, attr_emb='kg', seed=0):
        super().__init__()
        self.encoder = encoder
        self.use_gate = use_gate
        self.n_attr = n_attr
        if attr_emb == 'random':
            gen = torch.Generator().manual_seed(seed)
            emb = torch.randn(n_attr, d_h, generator=gen) * 0.1
            self.attr_emb = nn.Parameter(emb)
        else:
            # KG 属性向量：属性节点在向量空间中的可学习表示（由图谱属性集合初始化）
            self.attr_emb = nn.Parameter(torch.randn(n_attr, d_h) * 0.1)
        if use_gate:
            self.gate = GatingFusion(d_h)
        else:
            self.fuse = nn.Linear(2 * d_h, d_h)
        self.head = nn.Linear(d_h, 1)

    def forward(self, x):
        """返回 (B, n_attr) 的属性存在概率。"""
        h_d = self.encoder(x)                                  # (B, d_h)
        B = h_d.size(0)
        A = self.attr_emb.unsqueeze(0).expand(B, -1, -1).reshape(B * self.n_attr, -1)
        hd = h_d.unsqueeze(1).expand(B, self.n_attr, -1).reshape(B * self.n_attr, -1)
        if self.use_gate:
            h_f, _ = self.gate(hd, A)
        else:
            h_f = self.fuse(torch.cat([hd, A], dim=-1))
        return self.head(h_f).view(B, self.n_attr)


def dap_scores(p, M_cand):
    """DAP：log p(class) = Σ_j [a·log p_j + (1-a)·log(1-p_j)]，p: (B, n_attr)"""
    a = torch.from_numpy(M_cand).to(p.device)                  # (C, n_attr)
    eps = 1e-6
    lp = torch.log(p.clamp(eps, 1 - eps))                      # (B, n_attr)
    ln = torch.log((1 - p).clamp(eps, 1 - eps))                # (B, n_attr)
    # (B, C) = lp @ a.T + ln @ (1-a).T
    return lp @ a.t() + ln @ (1 - a).t()


def run_once(cfg, fold, k, seed, M_kg, epochs=200, batch=64, patience=40):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unseen_names = FOLDS[fold]
    unseen_ids = [CLASSES.index(c) for c in unseen_names]
    seen_ids = [i for i in range(NUM_CLASSES) if i not in unseen_ids]

    M = M_kg if cfg['matrix'] == 'kg' else permute_matrix(M_kg, seed)

    X, y = load_data()
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in seen_ids:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        tr_idx.extend(c_idx[:k].tolist())
        va_idx.extend(c_idx[k:k + 40].tolist())
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
    te_idx = np.concatenate([np.where(y == c)[0][-60:] for c in unseen_ids])

    enc = MLPEncoder(d_in=100, d_h=D_H)
    model = AttrZeroShotModel(enc, use_gate=cfg['gate'], attr_emb=cfg['attr_emb'],
                              seed=seed).to(device)
    set_protocol_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    A_tr = torch.from_numpy(M[y[tr_idx]]).float()              # (N_tr, n_attr)
    A_va = torch.from_numpy(M[y[va_idx]]).float().to(device)
    best_state, best_val, no_improve = None, 1e9, 0
    seen_local = {g: i for i, g in enumerate(seen_ids)}
    unseen_local = {g: i for i, g in enumerate(unseen_ids)}

    @torch.no_grad()
    def val_loss():
        """早停信号：可见类验证集上的属性 BCE 损失（越低越好）。

        注意：不能用 DAP 分类准确率，它在属性检测器收敛前长期停在随机水平，
        会导致「第 1 轮即最优」的假早停。
        """
        model.eval()
        xb = torch.from_numpy(X[va_idx]).float().to(device)
        return criterion(model(xb), A_va).item()

    @torch.no_grad()
    def eval_acc(idx, M_cand, y_local):
        model.eval()
        xb = torch.from_numpy(X[idx]).float().to(device)
        s = dap_scores(model(xb), M_cand)
        return (s.argmax(1).cpu().numpy() == y_local).mean()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(tr_idx), batch):
            b = perm[i:i + batch]
            xb = torch.from_numpy(X[tr_idx[b.numpy()]]).float().to(device)
            xb = xb + torch.randn_like(xb) * 0.02
            logits = model(xb)
            loss = criterion(logits, A_tr[b].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        v = val_loss()
        if v < best_val - 1e-5:
            best_val, no_improve = v, 0
            best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_te_local = np.array([unseen_local[int(v_)] for v_ in y[te_idx]])
    zs_acc = eval_acc(te_idx, M[unseen_ids], y_te_local)
    # 属性检测质量（可见类验证集，逐属性准确率）
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X[va_idx]).float().to(device)
        pred = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy()
    attr_acc = float((pred == M[y[va_idx]]).mean())
    return float(zs_acc), attr_acc, float(best_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-list', type=int, nargs='+', default=[20, 50])
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    attrs, M_kg = load_attribute_matrix()
    print('属性空间:', attrs)
    chance = 1.0 / 3

    results, attr_logs = {}, {}
    t0 = time.time()
    for k in args.k_list:
        for name, cfg in VARIANTS.items():
            accs, aas = [], []
            for fold in FOLDS:
                for seed in args.seeds:
                    acc, aacc, bv = run_once(cfg, fold, k, seed, M_kg, epochs=args.epochs)
                    accs.append(round(acc, 6)); aas.append(round(aacc, 6))
                    print(f'[k={k:2d} | {name:22s} | {fold} | seed={seed}] '
                          f'zs={acc:.4f} attr_acc={aacc:.3f} val={bv:.3f}', flush=True)
            results[f'{name}__k{k}'] = accs
            attr_logs[f'{name}__k{k}'] = aas

    from scipy.stats import ttest_rel
    print('\n' + '=' * 96)
    print(f'属性引导零样本未见故障诊断（3 折留出 × 5 种子，随机猜测水平 = {chance*100:.1f}%）')
    print('=' * 96)
    print(f'{"变体":<26}{"k":>4}{"零样本准确率":>20}{"属性检测率":>14}{"vs Ours":>14}{"p":>9}')
    print('-' * 96)
    summary = {'attributes': attrs, 'folds': FOLDS, 'chance': chance,
               'results': results, 'attr_acc': attr_logs, 'ttest': {}}
    for k in args.k_list:
        base = np.array(results[f'Ours-Attr (KG属性矩阵+门控)__k{k}'])
        for name in VARIANTS:
            a = np.array(results[f'{name}__k{k}'])
            aa = np.array(attr_logs[f'{name}__k{k}'])
            tag = ''
            if name != 'Ours-Attr (KG属性矩阵+门控)':
                t, p = ttest_rel(base, a)
                tag = f'{(a.mean()-base.mean())*100:+11.2f}pp{p:9.4f}'
                summary['ttest'][f'{name}__k{k}'] = {'delta': float(a.mean() - base.mean()),
                                                     'p': float(p), 't': float(t)}
            print(f'{name:<26}{k:>4}{a.mean()*100:14.2f}% ± {a.std()*100:4.2f}%{aa.mean()*100:12.2f}%{tag}')

    json.dump(summary, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n耗时 {time.time()-t0:.0f}s | 已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
