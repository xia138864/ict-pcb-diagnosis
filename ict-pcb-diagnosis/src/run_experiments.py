# -*- coding: utf-8 -*-
"""
SPICE 少样本诊断实验：MLP vs KG原型 vs KG门控融合 完整对比（早停版）
====================================================================
对多个 k × 多个 seed 跑三种模型，输出均值 ± 标准差 + 配对 t 检验。
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_data, few_shot_sample, SpiceDataset, AddNoise
from train import (build_model, train_with_early_stop, evaluate, stratified_split,
                   set_protocol_seed)

OUT_JSON = os.path.join(_ROOT, 'results_spice.json')


def run_once(model_type, k, seed, epochs=100, batch=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X, y = load_data()
    train_idx, val_idx, test_idx = stratified_split(y, seed=seed)

    if k > 0:
        fs = few_shot_sample(y[train_idx], k, seed=seed)
        train_idx = train_idx[fs]

    model = build_model(model_type, device).to(device)

    set_protocol_seed(seed)
    train_ds = SpiceDataset(X[train_idx], y[train_idx], transform=AddNoise(sigma=0.02))
    val_ds = SpiceDataset(X[val_idx], y[val_idx], transform=None)
    test_ds = SpiceDataset(X[test_idx], y[test_idx], transform=None)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    train_with_early_stop(model, train_loader, val_loader, device, epochs=epochs)
    return evaluate(model, test_loader, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-list', type=int, nargs='+', default=[5, 10, 20, 50])
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    models = ['mlp', 'kgproto', 'kg2']
    results = {}
    for k in args.k_list:
        for m in models:
            accs = []
            for seed in args.seeds:
                acc = run_once(m, k, seed, epochs=args.epochs)
                accs.append(round(acc, 6))
                print(f'[k={k:2d} {m:8s} seed={seed}] acc={acc:.4f}', flush=True)
            results[f'{m}_k{k}'] = accs

    from scipy.stats import ttest_rel
    print('\n' + '=' * 80)
    print(f'{"k":>4} | {"MLP mean±std":>18} | {"KGProto mean±std":>18} | {"KG gate mean±std":>18}')
    print('-' * 80)
    summary = {'results': results, 'ttest': {}}
    for k in args.k_list:
        m_ = np.array(results[f'mlp_k{k}'])
        g_ = np.array(results[f'kgproto_k{k}'])
        k_ = np.array(results[f'kg2_k{k}'])
        print(f'{k:4d} | {m_.mean():.4f}±{m_.std():.4f} | {g_.mean():.4f}±{g_.std():.4f} | {k_.mean():.4f}±{k_.std():.4f}')
        summary['ttest'][f'k{k}'] = {}
        for a_name, b_name in [('kgproto', 'mlp'), ('kg2', 'mlp'), ('kg2', 'kgproto')]:
            a = np.array(results[f'{a_name}_k{k}'])
            b = np.array(results[f'{b_name}_k{k}'])
            t, p = ttest_rel(a, b)
            summary['ttest'][f'k{k}'][f'{a_name}_vs_{b_name}'] = {
                'delta': float(a.mean() - b.mean()), 't': float(t), 'p': float(p)
            }
            print(f'  {a_name:8s} vs {b_name:5s}: Δ={a.mean()-b.mean():+.4f} p={p:.4f}')

    json.dump(summary, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
