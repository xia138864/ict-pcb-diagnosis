# -*- coding: utf-8 -*-
"""
数据分支：MLP 编码器（对应论文第 3.2 节）
=========================================
处理 100 点 ICT 测量向量：100 -> 128 -> 64（论文原设定 d_m=100, d_h=64）。
"""
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, d_in=100, d_hidden=128, d_h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_h),
        )

    def forward(self, x):
        """x: (B, 100) -> (B, 64)"""
        return self.net(x)
