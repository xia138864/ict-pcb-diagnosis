# -*- coding: utf-8 -*-
"""
扩展故障库（22 类，8 维粗粒度物理属性）
========================================
设计依据（由上一轮验证得出）：
1. 被测电路是强耦合网络，**节点级「受影响」属性不可用**——
   任一强故障都会扰动全网络（z 普遍 >10），AC 更是全局耦合（drift 类各节点 z≈5.2）。
   故放弃位置属性，改用可学习、可区分的**故障机理属性**。
2. 部分故障物理上不可测（如 cold_solder_R7 因 R8=100Ω 并联，节点 5 阻抗由 R8 主导，
   串联 1k 仅带来 ~1% 变化，z≈0.1），必须剔除。
3. 每个属性需有足够多**机理不同**的正例类别，属性检测器才能学到可迁移概念。

属性空间（8 维，全部由图谱遍历导出）：
  1 dc_shift        直流工作点偏移
  2 ac_shift        交流频响偏移
  3 current_anomaly 关键支路电流异常
  4 fluctuation     重复测量间波动
  5 topology_open   通路被切断（拓扑断连）
  6 ground_clamp    节点被钳位至地电位
  7 resistance_offset 电阻性参数偏移（非断连、非钳位）
  8 node_equalize   相邻节点等电位（桥接）
"""

# 现象 -> 属性（图谱 has_phenomenon -> manifests_as 的落点）
PHENOMENON_TO_MANIFEST = {
    'downstream nodes lose voltage':       ['dc_shift', 'current_anomaly', 'topology_open'],
    'node voltage pulled to ground':       ['dc_shift', 'current_anomaly', 'ground_clamp'],
    'filter frequency response shifted':   ['ac_shift', 'resistance_offset'],
    'missing filter capacitance':          ['ac_shift', 'topology_open'],
    'increased contact resistance':        ['dc_shift', 'resistance_offset'],
    'adjacent nodes electrically bridged': ['dc_shift', 'node_equalize', 'resistance_offset'],
    'fluctuating measurements':            ['fluctuation', 'resistance_offset'],
    'wrong voltage divider ratio':         ['dc_shift', 'resistance_offset'],
}

# (类别名, 仿真器故障规格, 现象)
FAULTS_EXT = [
    # --- 开路 4（拓扑断连）---
    ('open_R1',  {'type': 'R', 'name': 'R1', 'mode': 'open'},  'downstream nodes lose voltage'),
    ('open_R3',  {'type': 'R', 'name': 'R3', 'mode': 'open'},  'downstream nodes lose voltage'),
    ('open_R4',  {'type': 'R', 'name': 'R4', 'mode': 'open'},  'downstream nodes lose voltage'),
    ('open_R6',  {'type': 'R', 'name': 'R6', 'mode': 'open'},  'downstream nodes lose voltage'),
    # --- 短路到地 4（钳位）---
    ('short_R2', {'type': 'R', 'name': 'R2', 'mode': 'short'}, 'node voltage pulled to ground'),
    ('short_R5', {'type': 'R', 'name': 'R5', 'mode': 'short'}, 'node voltage pulled to ground'),
    ('short_R7', {'type': 'R', 'name': 'R7', 'mode': 'short'}, 'node voltage pulled to ground'),
    ('short_R10', {'type': 'R', 'name': 'R10', 'mode': 'short'}, 'node voltage pulled to ground'),
    # --- 电容参数漂移 3（频响偏移）---
    ('drift_C1', {'type': 'C', 'name': 'C1', 'mode': 'drift'}, 'filter frequency response shifted'),
    ('drift_C3', {'type': 'C', 'name': 'C3', 'mode': 'drift'}, 'filter frequency response shifted'),
    ('drift_C5', {'type': 'C', 'name': 'C5', 'mode': 'drift'}, 'filter frequency response shifted'),
    # --- 电容缺件 2 ---
    ('missing_C2', {'type': 'C', 'name': 'C2', 'mode': 'open'}, 'missing filter capacitance'),
    ('missing_C4', {'type': 'C', 'name': 'C4', 'mode': 'open'}, 'missing filter capacitance'),
    # --- 虚焊 3（串联接触电阻；R7 因 R8 并联不可测，已剔除）---
    ('cold_solder_R2', {'type': 'R', 'name': 'R2', 'mode': 'series'}, 'increased contact resistance'),
    ('cold_solder_R4', {'type': 'R', 'name': 'R4', 'mode': 'series'}, 'increased contact resistance'),
    ('cold_solder_R9', {'type': 'R', 'name': 'R9', 'mode': 'series'}, 'increased contact resistance'),
    # --- 桥接 3 ---
    ('bridge_3_4', {'type': 'B', 'a': 3, 'b': 4, 'mode': 'bridge'}, 'adjacent nodes electrically bridged'),
    ('bridge_4_5', {'type': 'B', 'a': 4, 'b': 5, 'mode': 'bridge'}, 'adjacent nodes electrically bridged'),
    ('bridge_5_6', {'type': 'B', 'a': 5, 'b': 6, 'mode': 'bridge'}, 'adjacent nodes electrically bridged'),
    # --- 间歇性 3 ---
    ('intermittent_R2', {'type': 'R', 'name': 'R2', 'mode': 'intermittent'}, 'fluctuating measurements'),
    ('intermittent_R4', {'type': 'R', 'name': 'R4', 'mode': 'intermittent'}, 'fluctuating measurements'),
    ('intermittent_R6', {'type': 'R', 'name': 'R6', 'mode': 'intermittent'}, 'fluctuating measurements'),
]

CLASS_NAMES_EXT = [c for c, _, _ in FAULTS_EXT]
FAULT_SPECS_EXT = {c: s for c, s, _ in FAULTS_EXT}
FAULT_PHEN_EXT = {c: p for c, _, p in FAULTS_EXT}

ATTRIBUTES = ['ac_shift', 'current_anomaly', 'dc_shift', 'fluctuation',
              'ground_clamp', 'node_equalize', 'resistance_offset', 'topology_open']

ATTRIBUTE_DESC = {
    'dc_shift':         '直流工作点偏移：节点直流电压显著偏离标称',
    'ac_shift':         '交流频响偏移：频响曲线形状或截止点改变',
    'current_anomaly':  '关键支路电流异常',
    'fluctuation':      '重复测量间数值波动',
    'topology_open':    '通路被切断（拓扑断连）',
    'ground_clamp':     '节点被钳位至地电位',
    'resistance_offset': '电阻性参数偏移（非断连、非钳位）',
    'node_equalize':    '相邻节点等电位（桥接）',
}


def class_attributes(cls):
    """类 -> 属性向量（由图谱现象层遍历导出）"""
    attrs = set(PHENOMENON_TO_MANIFEST[FAULT_PHEN_EXT[cls]])
    return {a: int(a in attrs) for a in ATTRIBUTES}


if __name__ == '__main__':
    from collections import Counter
    cov = Counter()
    for c in CLASS_NAMES_EXT:
        for a, v in class_attributes(c).items():
            cov[a] += v
    print(f'故障类别数: {len(CLASS_NAMES_EXT)}    属性维度: {len(ATTRIBUTES)}')
    print('\n各属性正例类别数（需 ≥3）：')
    for a in ATTRIBUTES:
        flag = 'OK ' if cov[a] >= 3 else 'LOW'
        print(f'  [{flag}] {a:<18} {cov[a]}')
    # 属性向量形态
    vecs = {}
    for c in CLASS_NAMES_EXT:
        v = tuple(sorted(a for a, x in class_attributes(c).items() if x))
        vecs.setdefault(v, []).append(c)
    print(f'\n不同属性向量形态（共 {len(vecs)} 种，用于构造可区分的留出折）：')
    for v, cs in vecs.items():
        print(f'  {set(v)}  ← {cs}')
