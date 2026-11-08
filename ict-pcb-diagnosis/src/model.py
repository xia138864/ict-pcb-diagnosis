# -*- coding: utf-8 -*-
"""
双分支整体模型（对应论文第 3 节）
================================
- 数据分支：MLP 编码器（100 点测量向量 -> h_d，对应论文 3.2 节）
- 知识分支：KG 子图池化 + 可学习投影 -> 类别知识原型 P[c]（对应论文 3.3 节）
- 门控融合：多维门控注意力（对应论文 3.4 节）
- 分类：原型网络式打分 / 门控融合 + MLP 打分

模型：MLPClassifier（纯数据基线）、KGProtoNet（无门控原型）、KGGuidedModel（门控融合）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gating import GatingFusion
from data_encoder import MLPEncoder
from dataset import CLASSES, NUM_CLASSES


class KGPrototype(nn.Module):
    """知识分支：预计算每类子图池化向量，经可学习投影 W_p 得到知识原型。"""
    def __init__(self, entity_emb, subgraph_idxs, d_k=128, d_h=64, freeze_emb=True):
        super().__init__()
        self.num_classes = len(CLASSES)
        self.d_h = d_h
        self.entity_emb = nn.Embedding.from_pretrained(entity_emb, freeze=freeze_emb)
        self.W_p = nn.Linear(d_k, d_h)

        pools = []
        for c in CLASSES:
            idxs = torch.tensor(subgraph_idxs[c], dtype=torch.long)
            emb = self.entity_emb(idxs)
            pools.append(emb.mean(dim=0))
        self.register_buffer('pools', torch.stack(pools))  # (8, d_k)

    def prototypes(self):
        return self.W_p(self.pools)  # (8, d_h)


class MLPClassifier(nn.Module):
    """纯数据基线：MLP -> 分类，无知识分支。"""
    def __init__(self, d_in=100, d_h=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = MLPEncoder(d_in=d_in, d_h=d_h)
        self.classifier = nn.Linear(d_h, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class KGProtoNet(nn.Module):
    """无门控原型网络：score = cos(h_d, P[c])，验证 KG 原型本身的价值。"""
    def __init__(self, encoder, kg_proto, scale=10.0):
        super().__init__()
        self.encoder = encoder
        self.kg_proto = kg_proto
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        h_d = F.normalize(self.encoder(x), dim=1)
        P = F.normalize(self.kg_proto.prototypes(), dim=1)
        return self.scale * (h_d @ P.t())


class KGGuidedModel(nn.Module):
    """完整模型：门控融合 + MLP 打分（relation-network 式）。"""
    def __init__(self, encoder, kg_proto, d_h=64):
        super().__init__()
        self.encoder = encoder
        self.kg_proto = kg_proto
        self.gate = GatingFusion(d_h)
        self.score_head = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(inplace=True),
            nn.Linear(d_h, 1),
        )

    def forward(self, x):
        h_d = self.encoder(x)                   # (B, d_h)
        P = self.kg_proto.prototypes()          # (8, d_h)
        B = h_d.size(0)
        scores = []
        for c in range(self.kg_proto.num_classes):
            p_c = P[c].unsqueeze(0).expand(B, -1)
            h_f, _ = self.gate(h_d, p_c)
            s_c = self.score_head(h_f)
            scores.append(s_c)
        return torch.cat(scores, dim=1)         # (B, 8)

    def gate_alpha(self, x):
        h_d = self.encoder(x)
        P = self.kg_proto.prototypes()
        B = h_d.size(0)
        alphas = []
        for c in range(self.kg_proto.num_classes):
            p_c = P[c].unsqueeze(0).expand(B, -1)
            _, alpha = self.gate(h_d, p_c)
            alphas.append(alpha.mean(dim=1))
        return torch.stack(alphas, dim=1)


class KGGuidedModelV2(nn.Module):
    """门控融合 v2：软检索知识 + 门控融合 + 单分类器（最贴近论文原意）。

    h_k = softmax(h_d·P/τ)·P   （软检索类别知识原型）
    h_f = gate(h_d, h_k)        （逐维门控融合，论文 Eq 3.1-3.3）
    logits = classifier(h_f)    （标准分类）
    """
    def __init__(self, encoder, kg_proto, d_h=64, tau=0.5):
        super().__init__()
        self.encoder = encoder
        self.kg_proto = kg_proto
        self.gate = GatingFusion(d_h)
        self.classifier = nn.Linear(d_h, NUM_CLASSES)
        self.tau = tau

    def forward(self, x):
        h_d = self.encoder(x)                        # (B, d_h)
        P = self.kg_proto.prototypes()               # (8, d_h)
        sim = torch.matmul(h_d, P.t()) / self.tau    # (B, 8)
        a = F.softmax(sim, dim=1)                    # (B, 8)
        h_k = torch.matmul(a, P)                     # (B, d_h)
        h_f, _ = self.gate(h_d, h_k)                 # (B, d_h)
        return self.classifier(h_f)                  # (B, 8)

    def gate_alpha(self, x):
        h_d = self.encoder(x)
        P = self.kg_proto.prototypes()
        sim = torch.matmul(h_d, P.t()) / self.tau
        a = F.softmax(sim, dim=1)
        h_k = torch.matmul(a, P)
        _, alpha = self.gate(h_d, h_k)
        return alpha.mean(dim=1)
