# -*- coding: utf-8 -*-
"""
消融实验（Ablation Study）
===========================
以完整模型 Ours（软检索知识 + 门控融合 + 分类器）为基准，逐一替换 / 移除关键组件，
量化每个组件的独立贡献。

公平性保证：所有变体共用同一随机协议（train.set_protocol_seed），
面对完全相同的测量噪声序列与 mini-batch 打乱顺序，差异只来自模型本身。

消融维度：
  A 融合机制   门控(完整) / 拼接 / 相加 / 固定 α=0.5
  B 知识信号   软检索(完整) / 硬检索 / 均匀平均
  C 知识分支   有(完整) / 无(纯 MLP) / 可学习类别原型(无 KG)
  D 子图范围   1 跳(完整) / 0 跳(仅故障节点) / 2 跳
  E 预训练     有(完整) / 无(随机嵌入)

用法：
  python ablation.py --k 10 --out ../results_ablation_k10.json
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

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (CLASSES, NUM_CLASSES, load_data, few_shot_sample,
                     SpiceDataset, AddNoise)
from model import KGPrototype
from data_encoder import MLPEncoder
from gating import GatingFusion
from train import (train_with_early_stop, evaluate, stratified_split,
                   set_protocol_seed)

KG_DIR = os.path.join(_ROOT, 'kg')
D_H = 64
D_K = 128


# ----------------------------------------------------------------------------
# 消融变体定义
# ----------------------------------------------------------------------------
VARIANTS = {
    'Ours (完整模型)':          dict(fusion='gate',   retr='soft',    subgraph='1hop'),
    'A1 拼接融合':              dict(fusion='concat', retr='soft',    subgraph='1hop'),
    'A2 相加融合':              dict(fusion='add',    retr='soft',    subgraph='1hop'),
    'A3 固定门控 α=0.5':        dict(fusion='fixed',  retr='soft',    subgraph='1hop'),
    'B1 硬检索知识':            dict(fusion='gate',   retr='hard',    subgraph='1hop'),
    'B2 均匀知识信号':          dict(fusion='gate',   retr='uniform', subgraph='1hop'),
    'C1 无知识分支(纯MLP)':     dict(no_kg=True),
    'C2 可学习类别原型(无KG)':  dict(fusion='gate',   retr='soft',    subgraph='1hop', learnable_proto=True),
    'D1 0 跳(仅故障节点)':       dict(fusion='gate',   retr='soft',    subgraph='0hop'),
    'D2 2 跳子图':              dict(fusion='gate',   retr='soft',    subgraph='2hop'),
    'E1 无预训练(随机嵌入)':     dict(fusion='gate',   retr='soft',    subgraph='1hop', emb='random'),
}


# ----------------------------------------------------------------------------
# 消融模型
# ----------------------------------------------------------------------------
class AblationModel(nn.Module):
    def __init__(self, encoder, kg_proto, cfg, d_h=D_H, tau=0.5):
        super().__init__()
        self.encoder = encoder
        self.kg_proto = kg_proto
        self.cfg = cfg
        self.tau = tau
        self.no_kg = cfg.get('no_kg', False)
        self.learnable_proto = cfg.get('learnable_proto', False)
        fusion = cfg.get('fusion')

        if self.no_kg:
            self.classifier = nn.Linear(d_h, NUM_CLASSES)
            return

        if self.learnable_proto:                      # 无 KG：8 个自由可学习类别原型
            self.free_pool = nn.Parameter(torch.randn(NUM_CLASSES, D_K) * 0.1)
            self.W_p_free = nn.Linear(D_K, d_h)

        if fusion == 'gate':
            self.gate = GatingFusion(d_h)
            self.classifier = nn.Linear(d_h, NUM_CLASSES)
        elif fusion == 'concat':
            self.fuse = nn.Linear(2 * d_h, d_h)
            self.classifier = nn.Linear(d_h, NUM_CLASSES)
        elif fusion in ('add', 'fixed'):
            self.classifier = nn.Linear(d_h, NUM_CLASSES)
        else:
            raise ValueError(fusion)

    def prototypes(self):
        if self.learnable_proto:
            return self.W_p_free(self.free_pool)       # (8, d_h)
        return self.kg_proto.prototypes()

    def _knowledge(self, h_d):
        P = self.prototypes()                          # (8, d_h)
        retr = self.cfg['retr']
        if retr == 'soft':
            sim = torch.matmul(h_d, P.t()) / self.tau
            a = F.softmax(sim, dim=1)
            return torch.matmul(a, P)
        elif retr == 'hard':
            idx = torch.matmul(h_d, P.t()).argmax(dim=1)
            return P[idx]
        elif retr == 'uniform':
            return P.mean(dim=0, keepdim=True).expand(h_d.size(0), -1)
        raise ValueError(retr)

    def forward(self, x):
        h_d = self.encoder(x)
        if self.no_kg:
            return self.classifier(h_d)

        h_k = self._knowledge(h_d)
        fusion = self.cfg['fusion']
        if fusion == 'gate':
            h_f, _ = self.gate(h_d, h_k)
        elif fusion == 'concat':
            h_f = self.fuse(torch.cat([h_d, h_k], dim=-1))
        elif fusion == 'add':
            h_f = h_d + h_k
        elif fusion == 'fixed':
            h_f = 0.5 * h_d + 0.5 * h_k
        return self.classifier(h_f)


# ----------------------------------------------------------------------------
# 子图原型
# ----------------------------------------------------------------------------
def build_subgraph_idxs(k_hop):
    data = json.load(open(os.path.join(KG_DIR, 'fault_graph_clean.json'), encoding='utf-8'))
    g = nx.DiGraph()
    for n in data['nodes']:
        g.add_node(n['id'], type=n.get('type', '?'))
    for e in data['edges']:
        g.add_edge(e['source'], e['target'])

    transe = torch.load(os.path.join(KG_DIR, 'transe_embeddings.pt'), weights_only=False)
    entity2idx = transe['entity2idx']
    mapping = json.load(open(os.path.join(KG_DIR, 'defect_kg_mapping.json'), encoding='utf-8'))

    out = {}
    for cls, fault_node in mapping['deep_pcb_classes'].items():
        if fault_node not in g:
            out[cls] = []
            continue
        nodes, frontier = {fault_node}, {fault_node}
        for _ in range(k_hop):
            nxt = set()
            for n in frontier:
                nxt.update(g.successors(n))
                nxt.update(g.predecessors(n))
            nodes.update(nxt)
            frontier = nxt
        out[cls] = sorted(entity2idx[e] for e in nodes if e in entity2idx)
    return out


def build_proto(cfg, cache, seed):
    key = (cfg['subgraph'], cfg.get('emb', 'pretrained'))
    if key not in cache:
        transe = torch.load(os.path.join(KG_DIR, 'transe_embeddings.pt'), weights_only=False)
        emb = transe['entity_emb']
        if cfg.get('emb') == 'random':
            gen = torch.Generator().manual_seed(seed)
            emb = torch.randn(emb.shape, generator=gen) * float(emb.std())
        idxs = build_subgraph_idxs({'0hop': 0, '1hop': 1, '2hop': 2}[cfg['subgraph']])
        cache[key] = (emb, idxs)
    emb, idxs = cache[key]
    return KGPrototype(emb, {c: idxs[c] for c in CLASSES}, d_k=D_K, d_h=D_H, freeze_emb=True)


# ----------------------------------------------------------------------------
# 单次运行
# ----------------------------------------------------------------------------
def run_once(cfg, k, seed, epochs=100, batch=64, proto_cache=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X, y = load_data()
    train_idx, val_idx, test_idx = stratified_split(y, seed=seed)
    if k > 0:
        train_idx = train_idx[few_shot_sample(y[train_idx], k, seed=seed)]

    # 模块创建顺序与 train.build_model 严格一致（encoder -> kg_proto -> gate -> classifier），
    # 保证「Ours」变体与主实验的 Ours 完全同源、结果可对齐。
    enc = MLPEncoder(d_in=100, d_h=D_H)
    proto = None
    if not cfg.get('no_kg') and not cfg.get('learnable_proto'):
        proto = build_proto(cfg, proto_cache, seed)
    model = AblationModel(enc, proto, cfg, d_h=D_H).to(device)

    set_protocol_seed(seed)                            # 统一噪声与打乱顺序
    train_ds = SpiceDataset(X[train_idx], y[train_idx], transform=AddNoise(sigma=0.02))
    val_ds = SpiceDataset(X[val_idx], y[val_idx], transform=None)
    test_ds = SpiceDataset(X[test_idx], y[test_idx], transform=None)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    train_with_early_stop(model, train_loader, val_loader, device, epochs=epochs)
    return evaluate(model, test_loader, device)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--variants', type=str, nargs='+', default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    names = args.variants if args.variants else list(VARIANTS.keys())
    out = args.out or os.path.join(_ROOT, f'results_ablation_k{args.k}.json')

    proto_cache = {}
    results = {}
    t0 = time.time()
    for name in names:
        cfg = VARIANTS[name]
        accs = []
        for seed in args.seeds:
            acc = run_once(cfg, args.k, seed, epochs=args.epochs, proto_cache=proto_cache)
            accs.append(round(acc, 6))
            print(f'[k={args.k:2d} | {name:24s} | seed={seed}] acc={acc:.4f}', flush=True)
        results[name] = accs

    from scipy.stats import ttest_rel
    base = np.array(results['Ours (完整模型)'])
    summary = {'k': args.k, 'seeds': args.seeds, 'results': results, 'ttest': {}}
    print('\n' + '=' * 92)
    print(f'消融实验  k = {args.k}（每类 {args.k} 个标注样本，{len(args.seeds)} 次运行）')
    print('=' * 92)
    print(f'{"变体":<26}{"准确率":>18}{"Δ vs Ours":>13}{"p":>9}')
    print('-' * 92)
    for name in names:
        a = np.array(results[name])
        if name == 'Ours (完整模型)':
            print(f'{name:<26}{a.mean()*100:8.2f}% ± {a.std()*100:5.2f}%{"—":>13}{"—":>9}')
            continue
        t, p = ttest_rel(base, a)
        d = a.mean() - base.mean()
        print(f'{name:<26}{a.mean()*100:8.2f}% ± {a.std()*100:5.2f}%{d*100:+12.2f}pp{p:9.4f}')
        summary['ttest'][name] = {'mean': float(a.mean()), 'std': float(a.std()),
                                  'delta': float(d), 't': float(t), 'p': float(p),
                                  'accs': results[name]}

    json.dump(summary, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n耗时 {time.time()-t0:.0f}s | 已保存: {out}')


if __name__ == '__main__':
    main()
