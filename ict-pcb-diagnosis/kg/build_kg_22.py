# -*- coding: utf-8 -*-
"""
构建 22 类故障库的知识图谱 + 属性矩阵
======================================
在清洗后的 ICT 图谱之上，为 22 类故障建立节点与关系：

  fault --is_type_of-->     故障族（open / short / failure / intermittent failure）
  fault --has_phenomenon--> 现象节点
  phenomenon --manifests_as--> 可观测表现（8 维物理属性）
  fault --has_component-->  故障元件节点 --connected_to--> 测试节点

说明：加入「故障元件」节点后，同类不同位置的故障（如 open_R1 与 open_R6）
子图不再相同，知识原型具备区分度。

输出：
  kg/fault_graph_clean.json   （覆盖：22 类故障库对应的图谱）
  kg/defect_kg_mapping.json   （类别 -> 故障节点）
  kg/class_attributes.json    （属性矩阵 + 溯源路径）
"""
import os
import sys
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fault_lib_ext import (CLASS_NAMES_EXT, FAULT_SPECS_EXT, FAULT_PHEN_EXT,
                           PHENOMENON_TO_MANIFEST, ATTRIBUTES, ATTRIBUTE_DESC,
                           class_attributes)  # noqa: E402

GRAPH_FILE = os.path.join(HERE, 'fault_graph_clean.json')
MAPPING_FILE = os.path.join(HERE, 'defect_kg_mapping.json')
ATTR_FILE = os.path.join(HERE, 'class_attributes.json')
CLEAN_SCRIPT = os.path.join(HERE, 'clean_graph.py')

TYPE_OF = {'open': 'open', 'short': 'short', 'drift': 'failure',
           'series': 'failure', 'wrong': 'failure', 'bridge': 'short',
           'intermittent': 'intermittent failure'}

# 元件 -> 两端测试节点
COMPONENT_NODES = {
    'R1': (1, 2), 'R2': (2, 0), 'R3': (2, 3), 'R4': (3, 4), 'R5': (4, 0),
    'R6': (4, 5), 'R7': (5, 0), 'R8': (6, 0), 'R9': (2, 7), 'R10': (7, 0),
    'C1': (3, 0), 'C2': (4, 0), 'C3': (5, 0), 'C4': (6, 0), 'C5': (7, 0),
}


def component_of(spec):
    if spec['mode'] == 'bridge':
        return f"bridge_{spec['a']}_{spec['b']}", (spec['a'], spec['b'])
    return spec['name'], COMPONENT_NODES[spec['name']]


def main():
    # 0) 备份并重建干净图谱
    for f in (GRAPH_FILE, MAPPING_FILE, ATTR_FILE):
        if os.path.exists(f):
            shutil.copy(f, f + '.8cls.bak')
    subprocess.run([sys.executable, CLEAN_SCRIPT], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL)

    g = json.load(open(GRAPH_FILE, encoding='utf-8'))
    nodes, edges = g['nodes'], g['edges']
    ids = {n['id'] for n in nodes}
    rels = {e['relation'] for e in edges}

    def add_node(nid, ntype):
        if nid not in ids:
            nodes.append({'id': nid, 'type': ntype}); ids.add(nid)

    def add_edge(s, r, t):
        e = {'source': s, 'relation': r, 'target': t}
        if e not in edges:
            edges.append(e)

    # 1) 现象 / 表现 / 故障族 / 测试节点
    for ph in PHENOMENON_TO_MANIFEST:
        add_node(ph, 'phenomenon')
    for a in ATTRIBUTES:
        add_node(a, 'manifestation')
        for ph, ms in PHENOMENON_TO_MANIFEST.items():
            if a in ms:
                add_edge(ph, 'manifests_as', a)
    for t in set(TYPE_OF.values()):
        add_node(t, 'fault_type')
    for n in range(1, 8):
        add_node(f'node_{n}', 'test_node')

    # 2) 22 类故障
    pre_n, pre_e = len(nodes), len(edges)
    for cls in CLASS_NAMES_EXT:
        spec = FAULT_SPECS_EXT[cls]
        add_node(cls, 'fault')
        add_edge(cls, 'is_type_of', TYPE_OF[spec['mode']])
        add_edge(cls, 'has_phenomenon', FAULT_PHEN_EXT[cls])
        comp, (na, nb) = component_of(spec)
        add_node(comp, 'component')
        for nd in {na, nb}:
            if nd != 0:
                add_edge(comp, 'connected_to', f'node_{nd}')
        add_edge(cls, 'has_component', comp)

    g['nodes'] = nodes; g['edges'] = edges
    json.dump(g, open(GRAPH_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # 3) 映射与属性矩阵
    mapping = {'deep_pcb_classes': {c: c for c in CLASS_NAMES_EXT},
               'fault_node': {c: c for c in CLASS_NAMES_EXT}}
    json.dump(mapping, open(MAPPING_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    provenance = {}
    for cls in CLASS_NAMES_EXT:
        ph = FAULT_PHEN_EXT[cls]
        comp, _ = component_of(spec := FAULT_SPECS_EXT[cls])
        provenance[cls] = {
            'fault_node': cls,
            'phenomenon': ph,
            'component': comp,
            'path': (f'{cls} -has_phenomenon-> {ph} -manifests_as-> '
                     + ', '.join(PHENOMENON_TO_MANIFEST[ph])),
        }
    json.dump({'attributes': ATTRIBUTES, 'attribute_desc': ATTRIBUTE_DESC,
               'class_attributes': {c: class_attributes(c) for c in CLASS_NAMES_EXT},
               'provenance': provenance},
              open(ATTR_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    n_fault = sum(1 for n in nodes if n['type'] == 'fault')
    print(f'图谱: {len(nodes)} 节点 / {len(edges)} 边 '
          f'（新增 {len(nodes)-pre_n} 节点、{len(edges)-pre_e} 边）')
    print(f'fault 节点: {n_fault}')
    print(f'关系类型: {sorted({e["relation"] for e in edges})}')
    print(f'属性维度: {len(ATTRIBUTES)}；映射类别数: {len(mapping["deep_pcb_classes"])}')
    print(f'已备份原 8 类产物为 *.8cls.bak')


if __name__ == '__main__':
    main()
