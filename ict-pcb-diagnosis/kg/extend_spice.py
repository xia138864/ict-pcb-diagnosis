# -*- coding: utf-8 -*-
"""
知识图谱扩充：为 SPICE 8 类电气故障建立独立的领域知识子图
========================================================
8 类故障各建独立 fault 节点，补充具体的"现象-原因"（有区分度），
并 is_type_of 关联到现有电气故障族（open/short/failure/intermittent）。
使 8 类各有独立的 2 跳子图，知识原型有区分度。

输出：更新 fault_graph_clean.json + defect_kg_mapping.json（8 类映射）
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(HERE, 'fault_graph_clean.json')
MAPPING_FILE = os.path.join(HERE, 'defect_kg_mapping.json')

# 8 类故障 -> 独立 fault 节点，含具体现象/原因（电气域，有区分度）
EXTENSIONS = {
    'open_circuit_R3': {
        'is_type_of': 'open',
        'has_phenomenon': ['downstream nodes lose voltage'],
        'caused_by': ['broken resistor R3'],
    },
    'short_circuit_R5': {
        'is_type_of': 'short',
        'has_phenomenon': ['node voltage pulled to ground'],
        'caused_by': ['shorted resistor R5'],
    },
    'capacitor_drift': {
        'is_type_of': 'failure',
        'has_phenomenon': ['filter frequency response shifted'],
        'caused_by': ['capacitor value drift'],
    },
    'missing_component': {
        'is_type_of': 'open',
        'has_phenomenon': ['missing filter capacitance'],
        'caused_by': ['component not populated'],
    },
    'cold_solder_joint': {
        'is_type_of': 'open',
        'has_phenomenon': ['increased contact resistance'],
        'caused_by': ['cold solder joint'],
    },
    'solder_bridge': {
        'is_type_of': 'short',
        'has_phenomenon': ['adjacent nodes electrically bridged'],
        'caused_by': ['solder bridge between traces'],
    },
    'intermittent_connection': {
        'is_type_of': 'intermittent failure',
        'has_phenomenon': ['fluctuating measurements'],
        'caused_by': ['loose connection'],
    },
    'wrong_component': {
        'is_type_of': 'failure',
        'has_phenomenon': ['wrong voltage divider ratio'],
        'caused_by': ['wrong component value'],
    },
}

# 故障类别 -> 独立 fault 节点
FAULT_NODE = {
    'open': 'open_circuit_R3',
    'short': 'short_circuit_R5',
    'drift': 'capacitor_drift',
    'missing': 'missing_component',
    'cold_solder': 'cold_solder_joint',
    'bridge': 'solder_bridge',
    'intermittent': 'intermittent_connection',
    'wrong_part': 'wrong_component',
}


def main():
    g = json.load(open(GRAPH_FILE, encoding='utf-8'))
    nodes = g['nodes']
    edges = g['edges']
    existing_ids = {n['id'] for n in nodes}

    new_nodes = []
    new_edges = []

    def add_node(nid, ntype):
        if nid not in existing_ids and not any(n['id'] == nid for n in new_nodes):
            new_nodes.append({'id': nid, 'type': ntype})
            existing_ids.add(nid)

    for fault, info in EXTENSIONS.items():
        add_node(fault, 'fault')
        new_edges.append({'source': fault, 'relation': 'is_type_of', 'target': info['is_type_of']})
        for r in info['caused_by']:
            add_node(r, 'reason')
            new_edges.append({'source': r, 'relation': 'caused_by', 'target': fault})
        for p in info['has_phenomenon']:
            add_node(p, 'phenomenon')
            new_edges.append({'source': fault, 'relation': 'has_phenomenon', 'target': p})

    nodes.extend(new_nodes)
    edges.extend(new_edges)
    g['nodes'] = nodes
    g['edges'] = edges
    json.dump(g, open(GRAPH_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # 更新映射：8 类 -> 独立 fault 节点
    mapping = {'deep_pcb_classes': FAULT_NODE, 'fault_node': FAULT_NODE}
    json.dump(mapping, open(MAPPING_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f'新增节点: {len(new_nodes)} 个')
    print(f'新增边: {len(new_edges)} 条')
    print(f'扩充后: {len(nodes)} 节点, {len(edges)} 边')
    print(f'8 类映射: {FAULT_NODE}')


if __name__ == '__main__':
    main()
