# -*- coding: utf-8 -*-
"""
知识图谱扩充：为 DeepPCB 6 类缺陷建立独立的领域知识子图
=====================================================
清洗后的 KG 只有 open/short 两族电气故障，无法区分 mousebite/spur/copper/pin-hole
这些 PCB 制造缺陷。本脚本为 4 个缺失的缺陷类别新增独立 fault 节点，并补充各自的
"现象-原因-元件"关系（PCB 制造领域知识），使 6 类各有独立的 2 跳子图，知识原型有区分度。

同时更新 defect_kg_mapping.json，将 6 类映射到各自的 fault 节点（6→6 而非 6→2）。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(HERE, 'fault_graph_clean.json')
MAPPING_FILE = os.path.join(HERE, 'defect_kg_mapping.json')

# 4 个新增缺陷类别的领域知识（PCB 制造常识）
# 关系方向与原 KG 一致：caused_by(原因→故障)、has_phenomenon(故障→现象)、
# is_type_of(具体→泛化)、related_to(双向相关)
EXTENSIONS = {
    'mousebite': {
        'desc': '板边铜箔被铣刀咬出缺口',
        'is_type_of': 'open',
        'caused_by': ['router bit offset', 'panel separation damage'],
        'has_phenomenon': ['copper missing at board edge'],
        'related_to': ['board edge trace'],
    },
    'spur': {
        'desc': '铜箔边缘突起的毛刺',
        'is_type_of': 'short',
        'caused_by': ['etching residue', 'overexposure'],
        'has_phenomenon': ['copper protrusion on trace'],
        'related_to': ['adjacent trace'],
    },
    'copper': {
        'desc': '不该有铜的位置出现多余铜箔',
        'is_type_of': 'short',
        'caused_by': ['film defect', 'incomplete etching'],
        'has_phenomenon': ['unwanted copper area'],
        'related_to': ['adjacent trace'],
    },
    'pin-hole': {
        'desc': '铜箔上的微小穿孔',
        'is_type_of': 'open',
        'caused_by': ['plating bubble', 'foreign particle'],
        'has_phenomenon': ['small void in copper'],
        'related_to': ['copper trace'],
    },
}


def main():
    g = json.load(open(GRAPH_FILE, encoding='utf-8'))
    nodes = g['nodes']
    edges = g['edges']
    existing_ids = {n['id'] for n in nodes}

    # 收集新增节点
    new_nodes = []
    new_edges = []

    def add_node(nid, ntype):
        if nid not in existing_ids and not any(n['id'] == nid for n in new_nodes):
            new_nodes.append({'id': nid, 'type': ntype})
            existing_ids.add(nid)

    for fault, info in EXTENSIONS.items():
        # fault 节点
        add_node(fault, 'fault')
        # is_type_of 父类
        new_edges.append({'source': fault, 'relation': 'is_type_of', 'target': info['is_type_of']})
        # caused_by 原因
        for r in info['caused_by']:
            add_node(r, 'reason')
            new_edges.append({'source': r, 'relation': 'caused_by', 'target': fault})
        # has_phenomenon 现象
        for p in info['has_phenomenon']:
            add_node(p, 'phenomenon')
            new_edges.append({'source': fault, 'relation': 'has_phenomenon', 'target': p})
        # related_to 元件
        for c in info['related_to']:
            add_node(c, 'component')
            new_edges.append({'source': fault, 'relation': 'related_to', 'target': c})

    nodes.extend(new_nodes)
    edges.extend(new_edges)
    g['nodes'] = nodes
    g['edges'] = edges
    json.dump(g, open(GRAPH_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # 更新 6 类映射：6 -> 6（各自 fault 节点）
    mapping = json.load(open(MAPPING_FILE, encoding='utf-8'))
    defect_classes = {
        'open': 'open',
        'short': 'short',
        'mousebite': 'mousebite',
        'spur': 'spur',
        'copper': 'copper',
        'pin-hole': 'pin-hole',
    }
    mapping['deep_pcb_classes'] = defect_classes
    json.dump(mapping, open(MAPPING_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f'新增节点: {len(new_nodes)} 个')
    print(f'新增边: {len(new_edges)} 条')
    print(f'扩充后: {len(nodes)} 节点, {len(edges)} 边')
    print(f'6 类映射更新: {defect_classes}')


if __name__ == '__main__':
    main()
