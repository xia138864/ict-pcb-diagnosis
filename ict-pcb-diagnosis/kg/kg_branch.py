# -*- coding: utf-8 -*-
"""
知识分支：子图原型提取
=======================
对 DeepPCB 6 类缺陷，各查询其 KG fault 节点的 2 跳邻域子图，
输出每类子图所涉实体的索引列表（供模型做均值池化 + 可学习投影，对应论文第 3.3 节）。

输出：
  kg/prototypes_subgraphs.pt   每类的 {fault_node, entity_idxs}
"""
import json
import os
import networkx as nx
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(HERE, 'fault_graph_clean.json')
TRANSE_FILE = os.path.join(HERE, 'transe_embeddings.pt')
MAPPING_FILE = os.path.join(HERE, 'defect_kg_mapping.json')
OUT_FILE = os.path.join(HERE, 'prototypes_subgraphs.pt')

K_HOP = 1


def load_graph():
    data = json.load(open(GRAPH_FILE, encoding='utf-8'))
    g = nx.DiGraph()
    for n in data['nodes']:
        g.add_node(n['id'], type=n.get('type', '?'))
    for e in data['edges']:
        g.add_edge(e['source'], e['target'], relation=e.get('relation', 'related_to'))
    return g


def get_khop_neighbors(g, node, k=K_HOP):
    """k 跳邻域实体集合（无向化：前驱 + 后继），含自身。"""
    if node not in g:
        return set()
    neighbors = {node}
    frontier = {node}
    for _ in range(k):
        nxt = set()
        for n in frontier:
            nxt.update(g.successors(n))
            nxt.update(g.predecessors(n))
        neighbors.update(nxt)
        frontier = nxt
    return neighbors


def main():
    g = load_graph()
    transe = torch.load(TRANSE_FILE, weights_only=False)
    entity2idx = transe['entity2idx']
    mapping = json.load(open(MAPPING_FILE, encoding='utf-8'))
    defect_classes = mapping['deep_pcb_classes']

    subgraphs = {}
    for cls, fault_node in defect_classes.items():
        neighbors = get_khop_neighbors(g, fault_node, k=K_HOP)
        idxs = sorted(entity2idx[e] for e in neighbors if e in entity2idx)
        subgraphs[cls] = {
            'fault_node': fault_node,
            'entity_idxs': idxs,
        }
        print(f'{cls:10s} -> fault [{fault_node:20s}] 子图实体数: {len(idxs)}')

    torch.save(subgraphs, OUT_FILE)
    print(f'子图原型已保存: {OUT_FILE}')


if __name__ == '__main__':
    main()
