# -*- coding: utf-8 -*-
"""
属性覆盖度标定实验（Fold Sweep）
=================================
目的：回答「零样本到底需要多少故障类别」。

思路：真正的约束不是类别总数，而是**每个属性在可见集中有多少个（机理不同的）正例类别**。
本脚本在现有 8 类故障库上枚举所有 3 类留出折，记录每折的：
  · min_cov：所有属性中「可见集正例类别数」的最小值
  · mean_cov：平均覆盖度
  · 零样本准确率
然后给出「准确率 vs 最小属性覆盖度」的经验曲线，用于外推所需规模。

用法：
  python zeroshot_attr_sweep.py --seeds 2024 2025 2026 --epochs 200
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASSES, NUM_CLASSES, load_data
from data_encoder import MLPEncoder
from gating import GatingFusion
from train import set_protocol_seed
from zeroshot_attr import (load_attribute_matrix, AttrZeroShotModel, dap_scores,
                           run_once)

OUT_JSON = os.path.join(_ROOT, 'results_coverage_sweep.json')
K = 20


def fold_stats(unseen_ids, M):
    seen_ids = [i for i in range(NUM_CLASSES) if i not in unseen_ids]
    cov = M[seen_ids].sum(axis=0)                      # 每个属性的可见正例数
    un = M[unseen_ids]
    distinct = len({tuple(r) for r in un}) == len(un)
    return seen_ids, cov, cov.min(), cov.mean(), distinct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026])
    ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()

    attrs, M = load_attribute_matrix()
    print('属性空间:', attrs)

    # 枚举所有 3 类留出折
    folds = []
    for combo in combinations(range(NUM_CLASSES), 3):
        unseen = list(combo)
        seen, cov, mn, mean_cov, distinct = fold_stats(unseen, M)
        folds.append({'unseen': unseen, 'seen': seen, 'cov': cov.tolist(),
                      'min_cov': int(mn), 'mean_cov': float(mean_cov),
                      'distinct': bool(distinct)})

    feasible = [f for f in folds if f['distinct']]
    print(f'全部 3 类留出折: {len(folds)} 个；未见类属性向量两两可分的折: {len(feasible)} 个')
    print(f'  其中 min_cov=0（存在属性在可见集中无正例）: {sum(1 for f in folds if f["min_cov"] == 0)} 个')
    print(f'  其中 min_cov≥1（属性全有正例）: {sum(1 for f in folds if f["min_cov"] >= 1)} 个')
    print(f'  其中 min_cov≥2: {sum(1 for f in folds if f["min_cov"] >= 2)} 个')
    print(f'  其中 min_cov≥3: {sum(1 for f in folds if f["min_cov"] >= 3)} 个')

    # 对可行折跑零样本（门控 + KG 属性矩阵）
    cfg = dict(gate=True, attr_emb='kg', matrix='kg')
    records = []
    t0 = time.time()
    for f in feasible:
        fold_name = 'F' + ''.join(str(i) for i in f['unseen'])
        # 临时注入 FOLDS
        import zeroshot_attr as Z
        Z.FOLDS[fold_name] = [CLASSES[i] for i in f['unseen']]
        accs = []
        for seed in args.seeds:
            acc, aacc, _ = run_once(cfg, fold_name, K, seed, M, epochs=args.epochs)
            accs.append(round(acc, 6))
        rec = {'fold': fold_name,
               'unseen': [CLASSES[i] for i in f['unseen']],
               'min_cov': f['min_cov'], 'mean_cov': f['mean_cov'],
               'cov': f['cov'], 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs))}
        records.append(rec)
        print(f"[{fold_name}] 未见={rec['unseen']} min_cov={f['min_cov']} mean_cov={f['mean_cov']:.1f} "
              f"→ zs={rec['acc_mean']*100:.1f}% ± {rec['acc_std']*100:.1f}%", flush=True)

    # 汇总：按 min_cov 分组
    print('\n' + '=' * 88)
    print(f'属性覆盖度 vs 零样本准确率（8 类库，3 类留出，k={K}，猜测线 33.3%）')
    print('=' * 88)
    print(f'{"min_cov":>8}{"折数":>6}{"平均零样本准确率":>20}{"折明细":>12}')
    print('-' * 88)
    grouped = {}
    for r in records:
        grouped.setdefault(r['min_cov'], []).append(r)
    curve = []
    for mn in sorted(grouped):
        v = np.mean([r['acc_mean'] for r in grouped[mn]])
        print(f'{mn:>8}{len(grouped[mn]):>6}{v*100:>17.2f}%')
        curve.append({'min_cov': mn, 'n_folds': len(grouped[mn]), 'acc_mean': float(v)})

    json.dump({'attributes': attrs, 'k': K, 'chance': 1 / 3,
               'n_all_folds': len(folds), 'n_feasible': len(feasible),
               'records': records, 'curve': curve},
              open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n耗时 {time.time()-t0:.0f}s | 已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
