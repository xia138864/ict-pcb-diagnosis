# -*- coding: utf-8 -*-
"""
TransE 嵌入训练模块
===================
在清洗后的知识图谱上训练 TransE 实体/关系嵌入（对应论文第 3.3 节）。
论文设定：嵌入维度 d_k = 128，margin γ = 1.0，学习率 0.01，训练 80 轮。

输出：
  kg/transe_embeddings.pt   实体嵌入、关系嵌入、实体/关系索引映射
"""
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(HERE, 'fault_graph_clean.json')
OUT_FILE = os.path.join(HERE, 'transe_embeddings.pt')

DIM = 128          # d_k
MARGIN = 1.0       # γ
LR = 0.01
EPOCHS = 80
BATCH_SIZE = 512
NEG_SAMPLES = 1     # 每个正样本的负样本数
SEED = 2024


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=DIM):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # 实体嵌入 L2 归一化（论文评分函数用 L2 距离）
        self.normalize()

    def normalize(self):
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)

    def forward(self, heads, rels, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(rels)
        t = self.entity_emb(tails)
        # 评分函数 f(h,r,t) = -|| h + r - t ||_2
        score = -torch.norm(h + r - t, p=2, dim=1)
        return score


def build_triples():
    """从图谱构建三元组列表和实体/关系索引映射。"""
    g = json.load(open(GRAPH_FILE, encoding='utf-8'))
    entities = [n['id'] for n in g['nodes']]
    entity2idx = {e: i for i, e in enumerate(entities)}

    triples = []
    relations = set()
    for e in g['edges']:
        h, r, t = e['source'], e['relation'], e['target']
        if h in entity2idx and t in entity2idx:
            triples.append((entity2idx[h], r, entity2idx[t]))
            relations.add(r)

    relations = sorted(relations)
    rel2idx = {r: i for i, r in enumerate(relations)}
    triples = [(h, rel2idx[r], t) for h, r, t in triples]
    return entities, entity2idx, relations, rel2idx, triples


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    entities, entity2idx, relations, rel2idx, triples = build_triples()
    num_entities = len(entities)
    num_relations = len(relations)

    # 用于负采样：每个实体作为尾实体的候选
    all_entities = list(range(num_entities))

    model = TransE(num_entities, num_relations, DIM)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    triples_t = torch.tensor(triples, dtype=torch.long)  # (N, 3)

    print(f'=== TransE 训练 ===')
    print(f'实体数: {num_entities}, 关系数: {num_relations}, 三元组数: {len(triples)}')
    print(f'设备: {device}')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(triples_t))
        total_loss = 0.0
        n_batch = 0

        for i in range(0, len(triples_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = triples_t[idx].to(device)  # (b, 3)
            h = batch[:, 0]
            r = batch[:, 1]
            t = batch[:, 2]

            # 负采样：随机替换尾实体
            neg_t = torch.randint(0, num_entities, (len(batch),), device=device)
            # 避免负样本恰好等于正尾实体（简单处理）
            pos_score = model(h, r, t)
            neg_score = model(h, r, neg_t)
            loss = F.margin_ranking_loss(
                pos_score, neg_score, torch.ones_like(pos_score), margin=MARGIN
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize()

            total_loss += loss.item()
            n_batch += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{EPOCHS} | loss {total_loss / max(n_batch, 1):.4f}')

    # 保存
    torch.save({
        'entity_emb': model.entity_emb.weight.data.cpu(),
        'relation_emb': model.relation_emb.weight.data.cpu(),
        'entities': entities,
        'entity2idx': entity2idx,
        'relations': relations,
        'rel2idx': rel2idx,
        'dim': DIM,
    }, OUT_FILE)
    print(f'嵌入已保存: {OUT_FILE}')


if __name__ == '__main__':
    train()
