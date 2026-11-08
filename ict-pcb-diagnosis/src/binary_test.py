# -*- coding: utf-8 -*-
"""
2 分类验证：开路族 vs 短路族
=============================
将 DeepPCB 6 类合并为 2 族（open/mousebite/pin-hole → 开路族；short/spur/copper → 短路族），
验证知识图谱（open/short 电气故障知识）在"粒度匹配"的任务上是否有价值。
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_patches, few_shot_sample
from image_encoder import CNNEncoder

KG = os.path.join(_ROOT, 'kg')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 6类 -> 2族映射
TO2 = {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0}  # open/mousebite/pin-hole=0, short/spur/copper=1


def run(k, seed, use_kg):
    torch.manual_seed(seed); np.random.seed(seed)
    train_p, train_l, test_p, test_l = load_patches()
    train_l2 = np.array([TO2[int(x)] for x in train_l])
    test_l2 = np.array([TO2[int(x)] for x in test_l])

    idxs = few_shot_sample(train_l, k, seed=seed)
    X = torch.from_numpy(train_p[idxs]).float().unsqueeze(1) / 255.
    y = torch.from_numpy(train_l2[idxs]).long()
    Xt = torch.from_numpy(test_p).float().unsqueeze(1) / 255.
    yt = torch.from_numpy(test_l2).long()

    # 数据增强：水平/垂直翻转
    X = torch.cat([X, torch.flip(X, dims=[3]), torch.flip(X, dims=[2])])
    y = torch.cat([y, y, y])

    cnn = CNNEncoder(d_h=64).to(device)
    if use_kg:
        transe = torch.load(os.path.join(KG, 'transe_embeddings.pt'), weights_only=False)
        sub = torch.load(os.path.join(KG, 'prototypes_subgraphs.pt'), weights_only=False)
        emb = transe['entity_emb']
        proto2 = torch.stack([
            emb[torch.tensor(sub['open']['entity_idxs'], dtype=torch.long)].mean(0),
            emb[torch.tensor(sub['short']['entity_idxs'], dtype=torch.long)].mean(0),
        ]).to(device)  # (2, 128)
        proj = nn.Linear(128, 64).to(device)
        opt = torch.optim.Adam(list(cnn.parameters()) + list(proj.parameters()), lr=1e-3)
    else:
        cls = nn.Linear(64, 2).to(device)
        opt = torch.optim.Adam(list(cnn.parameters()) + list(cls.parameters()), lr=1e-3)

    ce = nn.CrossEntropyLoss()
    for _ in range(60):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            xb, yb = X[b].to(device), y[b].to(device)
            h = cnn(xb)
            if use_kg:
                P = proj(proto2)
                logits = 10 * (F.normalize(h, dim=1) @ F.normalize(P, dim=1).t())
            else:
                logits = cls(h)
            loss = ce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    cnn.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(Xt), 512):
            xb = Xt[i:i + 512].to(device); yb = yt[i:i + 512].to(device)
            h = cnn(xb)
            if use_kg:
                P = proj(proto2)
                logits = 10 * (F.normalize(h, dim=1) @ F.normalize(P, dim=1).t())
            else:
                logits = cls(h)
            correct += (logits.argmax(1) == yb).sum().item(); total += len(yb)
    return correct / total


if __name__ == '__main__':
    for k in [5, 10, 20, 50]:
        c = [run(k, s, False) for s in [2024, 2025, 2026, 2027, 2028]]
        g = [run(k, s, True) for s in [2024, 2025, 2026, 2027, 2028]]
        print(f'2分类 k={k:2d}: CNN={np.mean(c):.4f}±{np.std(c):.4f}  '
              f'KG={np.mean(g):.4f}±{np.std(g):.4f}  Δ={np.mean(g)-np.mean(c):+.4f}')
