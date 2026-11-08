# -*- coding: utf-8 -*-
"""
SPICE 式电路仿真数据生成（MNA 求解器）
======================================
生成 ICT 风格的测量向量数据集：8 类故障 × 每类 400 样本。
每个样本含 100 点测量向量（DC 节点电压 + AC 频响幅值 + 支路电流），
注入 Monte Carlo 元件容差（±5%）与测量噪声（SNR=20dB），模拟真实产线测量。

故障类别对齐知识图谱 fault 节点（open/short/failure/missing/intermittent）。
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mna import Circuit, solve_dc, solve_ac

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(os.path.dirname(HERE), 'data', 'spice_dataset.npz')
OUT_META = os.path.join(os.path.dirname(HERE), 'data', 'spice_meta.json')

# ---------------------------------------------------------------------------
# 基础电路：模拟信号调理板（节点 0=地，1..7 为测试节点）
# 元件: (kind, name, node_a, node_b, nominal_value)
# ---------------------------------------------------------------------------
BASE_COMPONENTS = [
    ('R', 'R1', 1, 2, 1e3),      # 分压器
    ('R', 'R2', 2, 0, 1e3),
    ('R', 'R3', 2, 3, 2e3),
    ('C', 'C1', 3, 0, 1e-6),     # RC 滤波级 1
    ('R', 'R4', 3, 4, 1e3),
    ('R', 'R5', 4, 0, 3e3),
    ('C', 'C2', 4, 0, 0.1e-6),   # RC 滤波级 2
    ('R', 'R6', 4, 5, 500),
    ('R', 'R7', 5, 0, 2e3),
    ('C', 'C3', 5, 0, 10e-6),    # 大电容
    ('L', 'L1', 5, 6, 10e-3),    # 电感
    ('R', 'R8', 6, 0, 100),
    ('C', 'C4', 6, 0, 1e-6),
    ('R', 'R9', 2, 7, 1e3),      # 支路
    ('R', 'R10', 7, 0, 2e3),
    ('C', 'C5', 7, 0, 0.47e-6),
]

TEST_NODES = [1, 2, 3, 4, 5, 6, 7]      # DC 测试节点（7 个）
AC_NODES = [2, 3, 4, 5, 6, 7]           # AC 测试节点（6 个）
FREQS = np.logspace(2, 5, 15)           # 15 个对数频率 100Hz–100kHz
CURRENT_BRANCHES = [('R1', 1, 2), ('R5', 4, 0), ('R8', 6, 0)]  # 3 个支路电流

# 向量维度：7 (DC) + 6*15 (AC 幅值) + 3 (支路电流) = 100
DIM = 7 + 6 * 15 + 3

# ---------------------------------------------------------------------------
# 8 类故障定义（对齐 KG fault 节点）
# ---------------------------------------------------------------------------
FAULTS = {
    'open':         {'type': 'R', 'name': 'R3', 'mode': 'open'},       # 开路
    'short':        {'type': 'R', 'name': 'R5', 'mode': 'short'},      # 短路
    'drift':        {'type': 'C', 'name': 'C1', 'mode': 'drift'},      # 参数漂移
    'missing':      {'type': 'C', 'name': 'C2', 'mode': 'open'},       # 缺件
    'cold_solder':  {'type': 'R', 'name': 'R7', 'mode': 'series'},     # 虚焊
    'bridge':       {'type': 'B', 'a': 3, 'b': 4, 'mode': 'bridge'},   # 桥接短路
    'intermittent': {'type': 'R', 'name': 'R4', 'mode': 'intermittent'},  # 间歇性
    'wrong_part':   {'type': 'R', 'name': 'R2', 'mode': 'wrong'},      # 错件
}

FAULT_KG = {   # 故障类别 -> KG fault 节点
    'open': 'open',
    'short': 'short',
    'drift': 'failure',
    'missing': 'open',
    'cold_solder': 'open',
    'bridge': 'short',
    'intermittent': 'intermittent failure',
    'wrong_part': 'failure',
}

CLASS_NAMES = list(FAULTS.keys())


def apply_fault(components, fault_spec, rng):
    """注入故障，返回新的元件列表。"""
    comps = [list(c) for c in components]
    mode = fault_spec['mode']

    def find(kind, name):
        for c in comps:
            if c[0] == kind and c[1] == name:
                return c
        return None

    if mode == 'open':
        c = find(fault_spec['type'], fault_spec['name'])
        c[4] = 1e9  # 开路：极大电阻
    elif mode == 'short':
        c = find(fault_spec['type'], fault_spec['name'])
        c[4] = 1e-3  # 短路：极小电阻
    elif mode == 'drift':
        c = find(fault_spec['type'], fault_spec['name'])
        c[4] *= rng.uniform(1.5, 2.0)  # +50%~+100% 漂移
    elif mode == 'series':
        c = find(fault_spec['type'], fault_spec['name'])
        c[4] += 1e3  # 虚焊：串联 1k 接触电阻
    elif mode == 'wrong':
        c = find(fault_spec['type'], fault_spec['name'])
        c[4] *= 10.0  # 错件：10 倍标称值
    elif mode == 'bridge':
        comps.append(['R', 'R_bridge', fault_spec['a'], fault_spec['b'], 50.0])  # 桥接 50Ω
    elif mode == 'intermittent':
        c = find(fault_spec['type'], fault_spec['name'])
        c[4] *= rng.uniform(1.0, 3.0)  # 间歇性：随机漂移 1~3 倍
    return comps


def monte_carlo(components, rng):
    """非故障元件 ±5% 容差扰动。"""
    comps = [list(c) for c in components]
    for c in comps:
        kind = c[0]
        if kind in ('R', 'C', 'L') and c[1] != 'R_bridge':
            c[4] *= rng.normal(1.0, 0.05)  # 5% 容差
    return comps


def measure(components):
    """返回 100 维测量向量。"""
    # DC 工作点
    c_dc = Circuit()
    for kind, name, a, b, val in components:
        if kind == 'R':
            c_dc.add_resistor(name, a, b, val)
        elif kind == 'C':
            c_dc.add_capacitor(name, a, b, val)
        elif kind == 'L':
            c_dc.add_inductor(name, a, b, val)
    c_dc.add_vsource('V1', 1, 0, 5.0)
    v_dc = solve_dc(c_dc)
    feat_dc = [v_dc[n] for n in TEST_NODES]

    # AC 频响
    c_ac = Circuit()
    for kind, name, a, b, val in components:
        if kind == 'R':
            c_ac.add_resistor(name, a, b, val)
        elif kind == 'C':
            c_ac.add_capacitor(name, a, b, val)
        elif kind == 'L':
            c_ac.add_inductor(name, a, b, val)
    c_ac.add_vsource('V2', 1, 0, 1.0)  # AC 激励 1V @ node1
    feat_ac = []
    for f in FREQS:
        v_ac = solve_ac(c_ac, f)
        feat_ac.extend([abs(v_ac[n]) for n in AC_NODES])

    # 支路电流（DC）
    feat_cur = []
    for name, a, b in CURRENT_BRANCHES:
        for kind, nm, na, nb, val in components:
            if nm == name:
                i = abs(v_dc[a] - v_dc[b]) / max(val, 1e-6)
                feat_cur.append(i)
                break

    return np.array(feat_dc + feat_ac + feat_cur, dtype=np.float64)


def generate(n_per_class=400, seed=2024):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        spec = FAULTS[cls_name]
        for _ in range(n_per_class):
            comps = apply_fault(BASE_COMPONENTS, spec, rng)
            comps = monte_carlo(comps, rng)
            feat = measure(comps)
            X.append(feat)
            y.append(cls_idx)
    X = np.array(X)
    y = np.array(y, dtype=np.int64)

    # 加测量噪声（SNR=20dB）
    signal_std = X.std(axis=0, keepdims=True)
    noise = rng.normal(0.0, signal_std * 0.1, size=X.shape)
    X = X + noise

    # 保存
    np.savez_compressed(OUT_FILE, X=X, y=y)
    json.dump({'classes': CLASS_NAMES, 'fault_kg': FAULT_KG, 'dim': DIM},
              open(OUT_META, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f'=== SPICE 数据集生成完成 ===')
    print(f'样本数: {X.shape[0]}, 维度: {X.shape[1]}')
    for i, c in enumerate(CLASS_NAMES):
        print(f'  {c:14s} -> KG[{FAULT_KG[c]:20s}] 样本 {n_per_class}')
    print(f'已保存: {OUT_FILE}')


if __name__ == '__main__':
    generate()
