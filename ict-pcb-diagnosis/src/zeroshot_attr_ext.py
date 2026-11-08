# -*- coding: utf-8 -*-
"""
扩展故障库上的零样本标定实验
==============================
目的：验证「属性覆盖度（min_cov）是零样本能力的决定因素」，
     并据此确定**真正需要多少故障类别**。

流程：
1. 枚举 22 类故障库的 3 类留出折，计算每折的 min_cov（各属性在可见集中的正例类别数最小值）；
2. 按 min_cov 分层抽样（≥3 / =2 / =1），每层取若干折；
3. 依次跑 DAP 零样本（KG 属性矩阵 + 门控融合）与随机属性矩阵对照；
4. 输出「覆盖度 vs 零样本准确率」曲线。

用法：
  python zeroshot_attr_ext.py --k 20 --seeds 2024 2025 2026 --folds-per-level 6
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import time
import argparse
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kg'))

from dataset import SpiceDataset, AddNoise
from data_encoder import MLPEncoder
from gating import GatingFusion
from train import set_protocol_seed
from fault_lib_ext import ATTRIBUTES  # noqa: F401  (运行时由 class_attributes.json 覆盖)
from dataset import CLASSES as CLASS_NAMES_EXT, load_data as _load_data

ROOT = _ROOT
ATTR_FILE = os.path.join(ROOT, 'kg', 'class_attributes.json')
OUT_JSON = os.path.join(ROOT, 'results_zeroshot_attr_ext.json')
D_H, D_K = 64, 128
N_ATTR = 8
CHANCE = 1.0 / 3


def attribute_matrix():
    """从知识图谱导出的属性矩阵（kg/class_attributes.json）。"""
    global N_ATTR
    d = json.load(open(ATTR_FILE, encoding='utf-8'))
    attrs = d['attributes']
    N_ATTR = len(attrs)
    return (np.array([[d['class_attributes'][c][a] for a in attrs]
                      for c in CLASS_NAMES_EXT], dtype=np.float32), attrs)


def dap_scores(p, M_cand):
    a = torch.from_numpy(M_cand).to(p.device)
    eps = 1e-6
    lp = torch.log(p.clamp(eps, 1 - eps))
    ln = torch.log((1 - p).clamp(eps, 1 - eps))
    return lp @ a.t() + ln @ (1 - a).t()


class AttrModel(nn.Module):
    def __init__(self, encoder, n_attr, d_h=D_H):
        super().__init__()
        self.encoder = encoder
        self.n_attr = n_attr
        self.attr_emb = nn.Parameter(torch.randn(n_attr, d_h) * 0.1)
        self.gate = GatingFusion(d_h)
        self.head = nn.Linear(d_h, 1)

    def forward(self, x):
        h_d = self.encoder(x)
        B = h_d.size(0)
        A = self.attr_emb.unsqueeze(0).expand(B, -1, -1).reshape(B * self.n_attr, -1)
        hd = h_d.unsqueeze(1).expand(B, self.n_attr, -1).reshape(B * self.n_attr, -1)
        h_f, _ = self.gate(hd, A)
        return self.head(h_f).view(B, self.n_attr)


class ExtTrainer:
    """扩展库（22 类）上的 DAP 零样本训练/评估。"""

    def __init__(self, X, y, M):
        self.X, self.y, self.M = X, y, M
        self.n_cls = len(CLASS_NAMES_EXT)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self, unseen, k, seed, use_random_matrix=False, epochs=200,
            batch=128, patience=40):
        torch.manual_seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        M = self.M
        if use_random_matrix:
            rng = np.random.default_rng(seed)
            M = self.M[rng.permutation(self.n_cls)]

        seen = [i for i in range(self.n_cls) if i not in unseen]
        rng = np.random.default_rng(seed)
        tr, va = [], []
        for c in seen:
            ci = np.where(self.y == c)[0]; rng.shuffle(ci)
            tr.extend(ci[:k].tolist()); va.extend(ci[k:k + 40].tolist())
        tr, va = np.array(tr), np.array(va)
        te = np.concatenate([np.where(self.y == c)[0][-60:] for c in unseen])

        model = AttrModel(MLPEncoder(d_in=100, d_h=D_H), N_ATTR).to(self.device)
        set_protocol_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()
        A_tr = torch.from_numpy(M[self.y[tr]]).float()
        A_va = torch.from_numpy(M[self.y[va]]).float().to(self.device)

        @torch.no_grad()
        def val_loss():
            model.eval()
            return crit(model(torch.from_numpy(self.X[va]).float().to(self.device)), A_va).item()

        best, bv, ni = None, 1e9, 0
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), batch):
                b = perm[i:i + batch]
                xb = torch.from_numpy(self.X[tr[b.numpy()]]).float().to(self.device)
                xb = xb + torch.randn_like(xb) * 0.02
                loss = crit(model(xb), A_tr[b].to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
            v = val_loss()
            if v < bv - 1e-5:
                bv, ni = v, 0
                best = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
            else:
                ni += 1
                if ni >= patience:
                    break
        if best:
            model.load_state_dict(best)

        model.eval()
        ul = {g: i for i, g in enumerate(unseen)}
        y_te = np.array([ul[int(v)] for v in self.y[te]])
        with torch.no_grad():
            p_te = torch.sigmoid(model(torch.from_numpy(self.X[te]).float().to(self.device)))
            zs = float((dap_scores(p_te, M[unseen]).argmax(1).cpu().numpy() == y_te).mean())
            p_va = torch.sigmoid(model(torch.from_numpy(self.X[va]).float().to(self.device)))
            attr_acc = float(((p_va > 0.5).cpu().numpy() == M[self.y[va]]).mean())
        return zs, attr_acc


def fold_min_cov(M, unseen):
    seen = [i for i in range(len(CLASS_NAMES_EXT)) if i not in unseen]
    return int(M[seen].sum(axis=0).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026])
    ap.add_argument('--folds-per-level', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()

    M, attrs = attribute_matrix()
    print(f'故障类别数: {len(CLASS_NAMES_EXT)}   属性维度: {N_ATTR}')

    # 1) 枚举所有 3 类留出折，统计覆盖度分布
    all_folds = []
    for combo in combinations(range(len(CLASS_NAMES_EXT)), 3):
        un = list(combo)
        vecs = [tuple(M[i]) for i in un]
        if len(set(vecs)) != 3:          # 未见类属性向量必须两两可分
            continue
        all_folds.append((un, fold_min_cov(M, un)))
    from collections import Counter
    dist = Counter(mc for _, mc in all_folds)
    print(f'可用的 3 类留出折（未见类属性两两可分）: {len(all_folds)} 个')
    print('按 min_cov 分布:', dict(sorted(dist.items())))

    # 2) 分层抽样
    levels = sorted(dist.keys(), reverse=True)
    chosen = []
    rng = np.random.default_rng(0)
    for lv in levels:
        pool = [f for f, mc in all_folds if mc == lv]
        rng.shuffle(pool)
        chosen.extend([(f, lv) for f in pool[:args.folds_per_level]])
    print(f'抽样折数: {len(chosen)}（每层最多 {args.folds_per_level} 个）\n')

    # 3) 跑实验
    X, y = _load_data()
    trainer = ExtTrainer(X, y, M)
    records = []

    def dump_progress():
        """增量保存：即使进程被中途终止也不丢已完成的结果。"""
        curve = []
        for lv in sorted({r['min_cov'] for r in records}):
            rs = [r for r in records if r['min_cov'] == lv]
            curve.append({'min_cov': lv, 'n_folds': len(rs),
                          'kg_matrix': float(np.mean([r['acc'] for r in rs]) * 100),
                          'random_matrix': float(np.mean([r['acc_random_matrix'] for r in rs]) * 100)})
        json.dump({'n_classes': len(CLASS_NAMES_EXT), 'attributes': attrs,
                   'k': args.k, 'chance': CHANCE, 'fold_dist': dict(sorted(dist.items())),
                   'records': records, 'curve': curve},
                  open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    t0 = time.time()
    for idx, (un, lv) in enumerate(chosen):
        accs, rnd_accs, aaccs = [], [], []
        for seed in args.seeds:
            a, aa = trainer.run(un, args.k, seed, False, epochs=args.epochs)
            b, _ = trainer.run(un, args.k, seed, True, epochs=args.epochs)
            accs.append(a); rnd_accs.append(b); aaccs.append(aa)
        rec = {'unseen': [CLASS_NAMES_EXT[i] for i in un], 'min_cov': lv,
               'acc': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
               'acc_random_matrix': float(np.mean(rnd_accs)),
               'attr_acc': float(np.mean(aaccs))}
        records.append(rec)
        dump_progress()
        print(f"[{idx+1:2d}/{len(chosen)}] min_cov={lv} 未见={rec['unseen']} "
              f"→ KG矩阵 {rec['acc']*100:5.1f}% ± {rec['acc_std']*100:4.1f}% | "
              f"随机矩阵 {rec['acc_random_matrix']*100:5.1f}% | 属性检测 {rec['attr_acc']*100:.1f}%",
              flush=True)

    # 4) 汇总曲线
    print('\n' + '=' * 92)
    print(f'属性覆盖度 vs 零样本准确率（22 类库，3 类留出，k={args.k}，猜测线 {CHANCE*100:.1f}%）')
    print('=' * 92)
    print(f'{"min_cov":>8}{"折数":>6}{"KG矩阵":>14}{"随机矩阵":>12}{"差值":>12}')
    print('-' * 92)
    curve = []
    for lv in sorted({r['min_cov'] for r in records}):
        rs = [r for r in records if r['min_cov'] == lv]
        kg = np.mean([r['acc'] for r in rs]) * 100
        rd = np.mean([r['acc_random_matrix'] for r in rs]) * 100
        print(f'{lv:>8}{len(rs):>6}{kg:>12.1f}%{rd:>11.1f}%{kg-rd:>+11.1f}pp')
        curve.append({'min_cov': lv, 'n_folds': len(rs), 'kg_matrix': float(kg),
                      'random_matrix': float(rd)})

    json.dump({'n_classes': len(CLASS_NAMES_EXT), 'attributes': attrs,
               'k': args.k, 'chance': CHANCE, 'fold_dist': dict(sorted(dist.items())),
               'records': records, 'curve': curve},
              open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n耗时 {time.time()-t0:.0f}s | 已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
