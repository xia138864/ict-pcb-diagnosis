# -*- coding: utf-8 -*-
"""
SPICE 数据集加载 + 少样本采样
==============================
加载 spice_dataset.npz（22 类电气故障 × 100 维测量向量）。

22 类故障库（每个故障属性均有 ≥3 个正例类别，以支持零样本诊断）：
  open(4) short(4) drift(3) missing(2) cold_solder(3) bridge(3) intermittent(3)
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ['open_R1', 'open_R3', 'open_R4', 'open_R6',
           'short_R2', 'short_R5', 'short_R7', 'short_R10',
           'drift_C1', 'drift_C3', 'drift_C5',
           'missing_C2', 'missing_C4',
           'cold_solder_R2', 'cold_solder_R4', 'cold_solder_R9',
           'bridge_3_4', 'bridge_4_5', 'bridge_5_6',
           'intermittent_R2', 'intermittent_R4', 'intermittent_R6']
NUM_CLASSES = len(CLASSES)
DATA_FILE = os.path.join(_ROOT, 'data/spice_dataset.npz')


class AddNoise:
    """测量噪声增强（模拟重复测量）。"""
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma


class SpiceDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X   # (N, 100) float64
        self.y = y   # (N,) int64
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.y[idx])


def few_shot_sample(labels, k, seed=2024):
    """每类随机取 k 个样本，返回索引数组。"""
    rng = np.random.default_rng(seed)
    idxs = []
    for c in range(NUM_CLASSES):
        c_idx = np.where(labels == c)[0]
        if k >= len(c_idx):
            sel = c_idx
        else:
            sel = rng.choice(c_idx, size=k, replace=False)
        idxs.extend(sel.tolist())
    return np.array(idxs, dtype=np.int64)


def load_data():
    data = np.load(DATA_FILE)
    return data['X'], data['y']
