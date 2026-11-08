# -*- coding: utf-8 -*-
"""
零样本未见类别实验（Zero-Shot Unseen-Fault Diagnosis）
======================================================
核心问题：当某类故障**没有任何标注样本**时，纯数据方法不可能识别它。
此时唯一可用的信息来自知识图谱——这正是 KG 不可替代的场景。

协议：
  1. 从 8 类故障中划出若干「未见类别」，训练时完全剔除；
  2. 只用「可见类别」的 k 个标注样本/类训练模型；
  3. 测试时在「未见类别」的测试样本上分类，候选集 = 未见类别集合；
  4. 随机猜测水平 = 1 / |未见类别|。

对照（2×2 设计）：
  · 原型来源：KG 原型（TransE + 子图池化）  vs  随机原型（同形状随机向量）
  · 融合方式：逐维门控融合（Ours）          vs  无门控（直接 MLP 打分）

若「KG 原型」显著高于「随机原型」且高于随机猜测，则证明知识图谱在数据缺失时不可替代。

用法：
  python zeroshot.py --k-list 20 50 --seeds 2024 2025 2026 2027 2028
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
from model import KGPrototype
from data_encoder import MLPEncoder
from gating import GatingFusion
from train import train_with_early_stop, set_protocol_seed

KG_DIR = os.path.join(_ROOT, 'kg')
OUT_JSON = os.path.join(_ROOT, 'results_zeroshot.json')
D_H = 64
D_K = 128

# 三个类别留出折（每折 3 个未见类别，随机猜测水平 33.3%）
FOLDS = {
    'F1': ['open', 'drift', 'cold_solder'],
    'F2': ['short', 'missing', 'bridge'],
    'F3': ['intermittent', 'wrong_part', 'drift'],
}

VARIANTS = {
    'Ours (KG原型+门控)':   dict(proto='kg',     gate=True,  mode='rel'),
    'KG原型 + 无门控':      dict(proto='kg',     gate=False, mode='rel'),
    '随机原型 + 门控':      dict(proto='random', gate=True,  mode='rel'),
    '随机原型 + 无门控':    dict(proto='random', gate=False, mode='rel'),
    '余弦原型匹配 (KG)':    dict(proto='kg',     gate=False, mode='cos'),
    '余弦原型匹配 (随机)':  dict(proto='random', gate=False, mode='cos'),
    '门控+余弦 (KG)':       dict(proto='kg',     gate=True,  mode='cos'),
    '门控+余弦 (随机)':     dict(proto='random', gate=True,  mode='cos'),
}


class ZeroShotScorer(nn.Module):
    """类别无关打分器，支持两种可泛化到未见类别的打分方式：

    · mode='rel'：s_c = f([h_f(h_d, P_c); P_c])  共享关系网络打分
    · mode='cos'：s_c = scale · cos(h_d 或 h_f, P_c)  余弦原型匹配
    """
    def __init__(self, encoder, proto, d_h=D_H, use_gate=True, mode='rel'):
        super().__init__()
        self.encoder = encoder
        self.proto = proto
        self.use_gate = use_gate
        self.mode = mode
        if use_gate:
            self.gate = GatingFusion(d_h)
        if mode == 'rel':
            self.scorer = nn.Sequential(
                nn.Linear(2 * d_h, d_h),
                nn.ReLU(inplace=True),
                nn.Linear(d_h, 1),
            )
        else:
            self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, class_ids):
        h_d = self.encoder(x)                                   # (B, d_h)
        P = self.proto.prototypes()[class_ids]                  # (C, d_h)
        B, C = h_d.size(0), P.size(0)
        hd = h_d.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
        pe = P.unsqueeze(0).expand(B, C, -1).reshape(B * C, -1)
        h_f = hd
        if self.use_gate:
            h_f, _ = self.gate(hd, pe)

        if self.mode == 'rel':
            return self.scorer(torch.cat([h_f, pe], dim=-1)).view(B, C)
        # 余弦原型匹配：逐 (样本, 类别) 对做余弦相似
        h_f = h_f.view(B, C, -1)
        sim = (F.normalize(h_f, dim=-1) * F.normalize(P, dim=-1).unsqueeze(0)).sum(dim=-1)
        return self.scale * sim


def load_protos(kind, seed):
    """构造 8 类知识原型（KG 原型 / 随机原型）。"""
    transe = torch.load(os.path.join(KG_DIR, 'transe_embeddings.pt'), weights_only=False)
    subgraphs = torch.load(os.path.join(KG_DIR, 'prototypes_subgraphs.pt'), weights_only=False)
    emb = transe['entity_emb']
    if kind == 'random':
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(emb.shape, generator=gen) * float(emb.std())
    idxs = {c: subgraphs[c]['entity_idxs'] for c in CLASSES}
    return KGPrototype(emb, idxs, d_k=D_K, d_h=D_H, freeze_emb=True)


def run_once(cfg, fold_name, k, seed, epochs=100, batch=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unseen_names = FOLDS[fold_name]
    unseen_ids = [CLASSES.index(c) for c in unseen_names]
    seen_ids = [i for i in range(NUM_CLASSES) if i not in unseen_ids]

    X, y = load_data()
    rng = np.random.default_rng(seed)

    # 可见类别：每类抽 k 个训练样本 + 40 个验证样本（验证也在可见类内）
    tr_idx, va_idx = [], []
    for c in seen_ids:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        tr_idx.extend(c_idx[:k].tolist())
        va_idx.extend(c_idx[k:k + 40].tolist())
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)

    # 未见类别：只用测试样本（训练中完全不可见）
    te_idx = np.concatenate([np.where(y == c)[0][-60:] for c in unseen_ids])

    tr_ds = SpiceDataset(X[tr_idx], y[tr_idx], transform=AddNoise(sigma=0.02))
    va_ds = SpiceDataset(X[va_idx], y[va_idx])
    te_ds = SpiceDataset(X[te_idx], y[te_idx])
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)

    # 模块创建顺序与 train.build_model 一致：encoder -> proto -> gate -> scorer
    enc = MLPEncoder(d_in=100, d_h=D_H)
    proto = load_protos(cfg['proto'], seed)
    model = ZeroShotScorer(enc, proto, d_h=D_H, use_gate=cfg['gate'],
                           mode=cfg.get('mode', 'rel')).to(device)

    set_protocol_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练：候选集 = 可见类别
    seen_local = {g: i for i, g in enumerate(seen_ids)}
    y_tr_local = torch.tensor([seen_local[int(v)] for v in y[tr_idx]], dtype=torch.long)
    seen_t = torch.tensor(seen_ids, dtype=torch.long)
    best_state, best_val, no_improve = None, -1.0, 0

    @torch.no_grad()
    def val_acc():
        model.eval()
        xb = torch.from_numpy(X[va_idx]).float().to(device)
        yb = torch.tensor([seen_local[int(v)] for v in y[va_idx]]).to(device)
        return (model(xb, seen_t).argmax(1) == yb).float().mean().item()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(tr_idx), batch):
            b = perm[i:i + batch]
            xb = torch.from_numpy(X[tr_idx[b.numpy()]]).float().to(device)
            xb = xb + torch.randn_like(xb) * 0.02
            vb = y_tr_local[b].to(device)
            loss = criterion(model(xb, seen_t), vb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        v = val_acc()
        if v > best_val:
            best_val, no_improve = v, 0
            best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= 15:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试：候选集 = 未见类别（候选集内为局部索引，需映射回真实标签）
    model.eval()
    unseen_t = torch.tensor(unseen_ids, dtype=torch.long)
    seen_t2 = torch.tensor(seen_ids, dtype=torch.long)
    unseen_local = {g: i for i, g in enumerate(unseen_ids)}
    y_te_local = np.array([unseen_local[int(v)] for v in y[te_idx]])
    with torch.no_grad():
        xb = torch.from_numpy(X[te_idx]).float().to(device)
        pred = model(xb, unseen_t).argmax(1).cpu().numpy()
        zs_acc = float((pred == y_te_local).mean())
        # sanity check：同一打分器在可见类别测试集上的准确率（应接近 100%）
        seen_te = np.concatenate([np.where(y == c)[0][-60:] for c in seen_ids])
        y_seen_local = np.array([seen_local[int(v)] for v in y[seen_te]])
        xsb = torch.from_numpy(X[seen_te]).float().to(device)
        spred = model(xsb, seen_t2).argmax(1).cpu().numpy()
        seen_acc = float((spred == y_seen_local).mean())
    return zs_acc, seen_acc, float(best_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-list', type=int, nargs='+', default=[20, 50])
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    results, val_log, seen_log = {}, {}, {}
    t0 = time.time()
    for k in args.k_list:
        for name, cfg in VARIANTS.items():
            accs, saccs = [], []
            for fold in FOLDS:
                for seed in args.seeds:
                    acc, sacc, bv = run_once(cfg, fold, k, seed, epochs=args.epochs)
                    accs.append(round(acc, 6))
                    saccs.append(round(sacc, 6))
                    print(f'[k={k:2d} | {name:20s} | {fold} | seed={seed}] zs={acc:.4f} seen={sacc:.4f} val={bv:.3f}', flush=True)
            results[f'{name}__k{k}'] = accs
            seen_log[f'{name}__k{k}'] = saccs

    from scipy.stats import ttest_rel
    chance = 1.0 / 3
    PAIR = {'随机原型 + 门控': 'Ours (KG原型+门控)',
            '随机原型 + 无门控': 'KG原型 + 无门控',
            '余弦原型匹配 (随机)': '余弦原型匹配 (KG)',
            '门控+余弦 (随机)': '门控+余弦 (KG)'}
    summary = {'chance': chance, 'folds': FOLDS, 'results': results,
               'seen_class_test': seen_log, 'ttest': {}}
    print('\n' + '=' * 96)
    print(f'零样本未见类别诊断（3 折留出 × 5 种子，随机猜测水平 = {chance*100:.1f}%）')
    print('=' * 96)
    print(f'{"变体":<24}{"k":>4}{"未见类(零样本)":>20}{"可见类(sanity)":>18}{"KG-随机":>14}{"p":>9}')
    print('-' * 96)
    for k in args.k_list:
        for name in VARIANTS:
            a = np.array(results[f'{name}__k{k}'])
            sa = np.array(seen_log[f'{name}__k{k}'])
            tag = ''
            if name in PAIR:
                ctrl = np.array(results[f'{PAIR[name]}__k{k}'])
                t, p = ttest_rel(ctrl, a)
                tag = f'{(ctrl.mean()-a.mean())*100:+11.2f}pp{p:9.4f}'
                summary['ttest'][f'{name}__k{k}'] = {'delta': float(ctrl.mean() - a.mean()),
                                                     't': float(t), 'p': float(p)}
            print(f'{name:<24}{k:>4}{a.mean()*100:13.2f}% ± {a.std()*100:4.2f}%{sa.mean()*100:15.2f}%{tag}')

    json.dump(summary, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n耗时 {time.time()-t0:.0f}s | 已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
