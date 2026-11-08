# -*- coding: utf-8 -*-
"""
知识图谱清洗脚本
================
将 ICT_agent 的 LLM 自动抽取图谱（ict_fault_graph.json）清洗为可用于实验的规范图谱：
1. 归并语义重复的 fault 节点（short 族 / open 族 / failure 族 / intermittent 族 / stuck-at 族）
2. 规范关系类型（20 种 → 论文声明的 6 种）
3. 建立 DeepPCB 6 类缺陷与 KG fault 节点的映射

输出：
  kg/fault_graph_clean.json   清洗后的图谱（node_link 格式）
  kg/defect_kg_mapping.json   DeepPCB 6 类缺陷 → KG fault 节点映射
"""
import json
import os
from collections import defaultdict

SRC = os.environ.get('ICT_FAULT_GRAPH', os.path.join(_ROOT, '..', '..', '..', 'run_data', 'ict_fault_graph.json'))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_GRAPH = os.path.join(HERE, 'fault_graph_clean.json')
OUT_MAPPING = os.path.join(HERE, 'defect_kg_mapping.json')


# ---------------------------------------------------------------------------
# 1. fault 节点归并映射：规范名 -> 需要合并的原始节点名列表
# ---------------------------------------------------------------------------
MERGE_MAP = {
    'open': [
        'open', 'open circuit', 'open circuits', 'opens', 'open solder connections',
        'open solder joint', 'solder opens', 'unexpected opens', '开路', '开路跳线',
        '管脚焊接open', '判断是否有管脚焊接open的缺陷', 'circuit continuity broken',
        'discontinuity', 'break', 'broken', 'broken circuit', 'broken path', 'SW301 open',
    ],
    'short': [
        'short', 'short circuit', 'short circuits', 'unexpected short', 'unexpected shorts',
        'high impedance short', 'low impedance short', 'phantom short', '短路', '严重短路',
        '对 VCC 短路', '夹具短路', 'G1 and CLR short', 'Shorts S0 to GND or VCC', 'UUT 短路',
        'bus contention',
    ],
    'failure': [
        'failure', 'fault', 'faults', 'Failure Message', 'premature failure', '故障',
        '最终失效', '测量不正确', '超时失败', '不匹配错误', 'Analog 失败',
    ],
    'intermittent failure': [
        'intermittent', '间歇性失败', '间歇性故障',
    ],
    'stuck-at fault': [
        'stuck at', 'stuck at fault', 'stuck-at-0', 'stuck-at-1', 'transition fault',
        'pattern fault', '一个输入不变化', '一个输出不匹配',
    ],
}

# 反向映射：原始节点名 -> 规范名
MERGE_TARGET = {}
for canon, names in MERGE_MAP.items():
    for n in names:
        MERGE_TARGET[n] = canon


# ---------------------------------------------------------------------------
# 2. 关系类型规范化：原始 -> 规范（归并到 6 种）
# ---------------------------------------------------------------------------
REL_MAP = {
    'causes': 'causes',
    'caused_by': 'caused_by',
    'has_phenomenon': 'has_phenomenon',
    'is_type_of': 'is_type_of',
    'related_to': 'related_to',
    'solved_by': 'solved_by',
    # 噪声关系归并
    'includes': 'related_to',
    'has_reason': 'caused_by',
    'detects': 'related_to',
    'tests': 'related_to',
    'tests for': 'related_to',
    'indicates': 'has_phenomenon',
    'results in': 'causes',
    'introduces': 'causes',
    'affects': 'related_to',
    'driven by': 'caused_by',
    'simulate': 'related_to',
    'located_between': 'related_to',
    'defined_as': 'related_to',
    'is_protected_by': 'related_to',
}


# ---------------------------------------------------------------------------
# 3. DeepPCB 6 类缺陷 -> KG fault 族
#    图像缺陷本质：open/mousebite/pin-hole → 铜箔开路族；short/spur/copper → 铜箔短路族
# ---------------------------------------------------------------------------
DEFECT_CLASSES = {
    'open': 'open',
    'short': 'short',
    'mousebite': 'open',    # 鼠咬：铜箔被咬缺 → 开路风险
    'spur': 'short',        # 毛刺：铜箔突起 → 桥接短路风险
    'copper': 'short',      # 多余铜 → 桥接短路风险
    'pin-hole': 'open',     # 针孔：铜箔穿孔 → 局部开路
}


def main():
    src = json.load(open(SRC, encoding='utf-8'))
    raw_nodes = src['nodes']
    raw_edges = src.get('edges', src.get('links', []))

    # 节点 id -> type
    id2type = {n['id']: n.get('type', '?') for n in raw_nodes}

    # ---- 归并节点 ----
    new_nodes = {}
    for n in raw_nodes:
        nid = n['id']
        ntype = n.get('type', '?')
        # fault 节点且命中归并表 -> 映射到规范名
        if ntype == 'fault' and nid in MERGE_TARGET:
            canon = MERGE_TARGET[nid]
            new_nodes[canon] = {'id': canon, 'type': 'fault'}
        else:
            new_nodes[nid] = {'id': nid, 'type': ntype}

    # ---- 重映射边 + 规范关系 + 去重 ----
    seen_edges = set()
    new_edges = []
    for e in raw_edges:
        s = e.get('source')
        t = e.get('target')
        rel = e.get('relation', e.get('predicate', 'related_to'))

        # 归并 source / target
        if s in MERGE_TARGET:
            s = MERGE_TARGET[s]
        if t in MERGE_TARGET:
            t = MERGE_TARGET[t]

        # 两端节点必须都存在
        if s not in new_nodes or t not in new_nodes:
            continue
        # 自环（归并后同一节点）丢弃
        if s == t:
            continue

        rel = REL_MAP.get(rel, 'related_to')

        key = (s, rel, t)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        new_edges.append({'source': s, 'relation': rel, 'target': t})

    # 统计
    node_type_count = defaultdict(int)
    for n in new_nodes.values():
        node_type_count[n['type']] += 1
    rel_count = defaultdict(int)
    for e in new_edges:
        rel_count[e['relation']] += 1

    clean_graph = {
        'directed': True,
        'multigraph': False,
        'graph': {},
        'nodes': list(new_nodes.values()),
        'edges': new_edges,
    }
    json.dump(clean_graph, open(OUT_GRAPH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # ---- 6 类缺陷映射 ----
    mapping = {
        'deep_pcb_classes': DEFECT_CLASSES,
        'kg_fault_nodes': {c: {'fault_node': fn, 'merged_from': MERGE_MAP.get(fn, [fn])}
                           for c, fn in DEFECT_CLASSES.items()},
        'merge_map': MERGE_MAP,
    }
    json.dump(mapping, open(OUT_MAPPING, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # 打印摘要
    print('=== 清洗完成 ===')
    print(f'原始: {len(raw_nodes)} 节点, {len(raw_edges)} 边')
    print(f'清洗后: {len(new_nodes)} 节点, {len(new_edges)} 边')
    print(f'节点类型分布: {dict(node_type_count)}')
    print(f'关系类型分布: {dict(rel_count)}')
    print(f'fault 节点归并: {len(MERGE_TARGET)} -> {len(MERGE_MAP)} 个规范节点')
    print(f'DeepPCB 6 类映射: {DEFECT_CLASSES}')


if __name__ == '__main__':
    main()
