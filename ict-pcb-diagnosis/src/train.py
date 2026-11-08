# -*- coding: utf-8 -*-
"""
训练 + 评估脚本（SPICE 数据版，含验证集早停）
==============================================
  python train.py --model mlp|kgproto|kg --k <每类样本数> --seed <种子>

  --model mlp      纯 MLP 数据基线
  --model kgproto  无门控 KG 原型网络
  --model kg       MLP + KG 门控融合（完整模型）
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (CLASSES, NUM_CLASSES, load_data, few_shot_sample,
                     SpiceDataset, AddNoise)
from model import KGGuidedModel, KGPrototype, MLPClassifier, KGProtoNet, KGGuidedModelV2
from data_encoder import MLPEncoder

KG_DIR = os.path.join(_ROOT, 'kg')
D_H = 64
D_K = 128


def _load_kg_proto():
    transe = torch.load(os.path.join(KG_DIR, 'transe_embeddings.pt'), weights_only=False)
    subgraphs = torch.load(os.path.join(KG_DIR, 'prototypes_subgraphs.pt'), weights_only=False)
    entity_emb = transe['entity_emb']
    subgraph_idxs = {c: subgraphs[c]['entity_idxs'] for c in CLASSES}
    return KGPrototype(entity_emb, subgraph_idxs, d_k=D_K, d_h=D_H, freeze_emb=True)


def build_model(model_type, device):
    if model_type == 'mlp':
        return MLPClassifier(d_in=100, d_h=D_H, num_classes=NUM_CLASSES)
    elif model_type == 'kgproto':
        return KGProtoNet(MLPEncoder(d_in=100, d_h=D_H), _load_kg_proto())
    elif model_type == 'kg':
        return KGGuidedModel(MLPEncoder(d_in=100, d_h=D_H), _load_kg_proto(), d_h=D_H)
    elif model_type == 'kg2':
        return KGGuidedModelV2(MLPEncoder(d_in=100, d_h=D_H), _load_kg_proto(), d_h=D_H)
    else:
        raise ValueError(f'未知模型类型: {model_type}')


def set_protocol_seed(seed):
    """统一实验协议：在构建 DataLoader 之前重置随机流。

    这样不同模型 / 消融变体面对完全相同的测量噪声序列与 mini-batch 打乱顺序，
    差异只来自模型本身，保证消融对比的公平性。
    """
    torch.manual_seed(seed + 999)
    np.random.seed(seed + 999)


def stratified_split(y, n_train_per_class=280, n_val_per_class=60, seed=2024):
    """按类分层划分 train/val/test。"""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(NUM_CLASSES):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        train_idx.extend(c_idx[:n_train_per_class].tolist())
        val_idx.extend(c_idx[n_train_per_class:n_train_per_class + n_val_per_class].tolist())
        test_idx.extend(c_idx[n_train_per_class + n_val_per_class:].tolist())
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train_with_early_stop(model, train_loader, val_loader, device, epochs=100,
                          lr=1e-3, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='kg', choices=['mlp', 'kgproto', 'kg', 'kg2'])
    ap.add_argument('--k', type=int, default=0, help='少样本每类样本数，0=全量')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'=== {args.model} | k={args.k} | seed={args.seed} | device={device} ===')

    X, y = load_data()
    train_idx, val_idx, test_idx = stratified_split(y, seed=args.seed)

    # 少样本采样（从 train 每类取 k 个）
    if args.k > 0:
        fs = few_shot_sample(y[train_idx], args.k, seed=args.seed)
        train_idx = train_idx[fs]
        print(f'少样本训练: 每类 {args.k} 个，共 {len(train_idx)} 样本')

    model = build_model(args.model, device).to(device)

    set_protocol_seed(args.seed)
    train_ds = SpiceDataset(X[train_idx], y[train_idx], transform=AddNoise(sigma=0.02))
    val_ds = SpiceDataset(X[val_idx], y[val_idx], transform=None)
    test_ds = SpiceDataset(X[test_idx], y[test_idx], transform=None)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    t0 = time.time()
    best_val = train_with_early_stop(model, train_loader, val_loader, device, args.epochs)
    acc = evaluate(model, test_loader, device)
    dt = time.time() - t0
    print(f'>>> 验证集最优: {best_val:.4f} | 测试准确率: {acc:.4f} ({acc*100:.2f}%) | 耗时 {dt:.1f}s')
    return acc


if __name__ == '__main__':
    main()
