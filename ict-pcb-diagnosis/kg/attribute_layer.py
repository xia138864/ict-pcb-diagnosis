# -*- coding: utf-8 -*-
"""
知识图谱属性层（manifestation layer）
=====================================
问题：原图谱中每个故障只有 1 个专属现象节点，映射到属性后是「对角矩阵」，
属性无法跨故障共享，模型学不到可迁移的「故障 → 测量表现」映射，零样本必然失败。

解决：在现象层之下再引入一层**可观测表现**（manifestation）节点，
由现象经 `manifests_as` 关系指向。多个故障会共享同一表现节点，
从而使属性矩阵具有重叠结构（非对角），属性分类器可在可见故障上训练并迁移到未见故障。

图谱路径：fault --has_phenomenon--> phenomenon --manifests_as--> manifestation

输出：
  kg/fault_graph_clean.json    （追加 manifestation 节点与 manifests_as 边）
  kg/class_attributes.json     （8 类 × 6 属性的矩阵 + 属性名 + 来源路径）
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(HERE, 'fault_graph_clean.json')
MAPPING_FILE = os.path.join(HERE, 'defect_kg_mapping.json')
OUT_FILE = os.path.join(HERE, 'class_attributes.json')

# 现象 -> 可观测表现（物理可测量的属性）
# 依据电路物理：每个现象会在 DC 工作点 / AC 频响 / 支路电流上留下特征。
# 除「性质」（nature）外，再引入「作用范围」（scope）与「频段」（band）两层，
# 使 8 类故障在属性空间中两两可分（否则 cold_solder 与 wrong_part 会完全重合）。
PHENOMENON_TO_MANIFEST = {
    'downstream nodes lose voltage':       ['voltage_loss', 'current_anomaly', 'scope_global'],
    'node voltage pulled to ground':       ['ground_clamp', 'current_anomaly', 'scope_local'],
    'filter frequency response shifted':   ['freq_shift', 'freq_low'],
    'missing filter capacitance':          ['freq_shift', 'freq_high'],
    'increased contact resistance':        ['voltage_loss', 'ratio_shift', 'scope_local'],
    'adjacent nodes electrically bridged': ['ground_clamp', 'ratio_shift', 'scope_local'],
    'fluctuating measurements':            ['fluctuation', 'ratio_shift', 'scope_local'],
    'wrong voltage divider ratio':         ['ratio_shift', 'voltage_loss', 'scope_global'],
}

MANIFEST_DESC = {
    # 性质层：异常在电气量上的表现类型
    'voltage_loss':    '直流节点电压偏离标称（失压/偏移）',
    'ground_clamp':    '某节点被钳位至地电位',
    'freq_shift':      '交流频响曲线形状/截止点偏移',
    'ratio_shift':     '分压比例改变导致下游电压重分配',
    'current_anomaly': '关键支路电流异常',
    'fluctuation':     '重复测量间数值波动',
    # 作用范围层
    'scope_global':    '多节点/整段下游同时异常（分布性故障）',
    'scope_local':     '异常集中于单点或相邻两点（局部性故障）',
    # 频段层
    'freq_low':        '低频段（百赫兹量级）频响改变',
    'freq_high':       '高频段（千赫兹量级）频响改变',
}


def main():
    g = json.load(open(GRAPH_FILE, encoding='utf-8'))
    mapping = json.load(open(MAPPING_FILE, encoding='utf-8'))
    nodes, edges = g['nodes'], g['edges']
    node_ids = {n['id'] for n in nodes}

    # 1. 追加 manifestation 节点与 manifests_as 边
    added_nodes, added_edges = 0, 0
    for ph, manifests in PHENOMENON_TO_MANIFEST.items():
        if ph not in node_ids:
            print(f'  [警告] 现象节点不存在: {ph}')
            continue
        for m in manifests:
            if m not in node_ids:
                nodes.append({'id': m, 'type': 'manifestation'})
                node_ids.add(m)
                added_nodes += 1
            e = {'source': ph, 'relation': 'manifests_as', 'target': m}
            if e not in edges:
                edges.append(e)
                added_edges += 1

    json.dump(g, open(GRAPH_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # 2. 从图谱推导 类 -> 属性集（两跳：has_phenomenon -> manifests_as）
    out_ph = {}
    out_man = {}
    for e in edges:
        if e['relation'] == 'has_phenomenon':
            out_ph.setdefault(e['source'], []).append(e['target'])
    for e in edges:
        if e['relation'] == 'manifests_as':
            out_man.setdefault(e['source'], []).append(e['target'])

    attributes = sorted(MANIFEST_DESC.keys())
    class_attr, provenance = {}, {}
    for cls, fault_node in mapping['deep_pcb_classes'].items():
        phs = out_ph.get(fault_node, [])
        ms = sorted({m for ph in phs for m in out_man.get(ph, [])})
        class_attr[cls] = {a: int(a in ms) for a in attributes}
        provenance[cls] = {
            'fault_node': fault_node,
            'phenomena': phs,
            'path': [f'{fault_node} -has_phenomenon-> {ph} -manifests_as-> '
                     + ', '.join(out_man.get(ph, [])) for ph in phs],
        }

    json.dump({'attributes': attributes, 'attribute_desc': MANIFEST_DESC,
               'class_attributes': class_attr, 'provenance': provenance},
              open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f'新增 manifestation 节点: {added_nodes}，新增 manifests_as 边: {added_edges}')
    print(f'属性空间（{len(attributes)} 维）: {attributes}')
    print('\n=== 8 类故障 × 属性矩阵 ===')
    print(f'{"类别":<14}' + ''.join(f'{a[:13]:>15}' for a in attributes))
    for cls, row in class_attr.items():
        print(f'{cls:<14}' + ''.join(f'{row[a]:>15}' for a in attributes))
    print('\n=== 属性覆盖（每属性出现的类别） ===')
    for a in attributes:
        cs = [c for c, r in class_attr.items() if r[a]]
        print(f'  {a:<16} {cs}')


if __name__ == '__main__':
    main()
