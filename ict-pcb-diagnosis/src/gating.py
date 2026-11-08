# -*- coding: utf-8 -*-
"""
多维门控注意力融合（论文第 3.4 节，核心创新模块）
==================================================
Eq 3.1:  h_d' = LN(h_d), h_k' = LN(h_k)
Eq 3.2:  α = σ(W_g [h_d'; h_k'] + b_g)         逐维门控权重
Eq 3.3:  h_f = α ⊙ h_d' + (1 - α) ⊙ h_k'       逐维凸组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingFusion(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.d_h = d_h
        self.W_g = nn.Linear(2 * d_h, d_h)
        # 初始化：偏向 α≈0.5（先均衡利用两分支）
        nn.init.zeros_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def forward(self, h_d, h_k):
        """h_d, h_k: (B, d_h)"""
        h_d_n = F.layer_norm(h_d, (self.d_h,))
        h_k_n = F.layer_norm(h_k, (self.d_h,))
        cat = torch.cat([h_d_n, h_k_n], dim=-1)          # (B, 2d_h)
        alpha = torch.sigmoid(self.W_g(cat))             # (B, d_h)
        h_f = alpha * h_d_n + (1.0 - alpha) * h_k_n       # (B, d_h)
        return h_f, alpha
