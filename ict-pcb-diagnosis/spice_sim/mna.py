# -*- coding: utf-8 -*-
"""
改进节点分析（MNA）电路求解器
==============================
SPICE 的核心算法。支持电阻、电容、电感、独立电压源/电流源，
做 DC 工作点分析与 AC 频响分析。对线性电路精确求解（基尔霍夫定律）。

MNA 矩阵形式：
  [ G   B ] [ v ]   [ I ]
  [ B'  0 ] [ i ] = [ E ]
其中 G 为导纳矩阵，B 为电压源连接矩阵，v 为节点电压，i 为电压源电流。
"""
import numpy as np


class Circuit:
    def __init__(self):
        self.components = []   # (kind, name, a, b, value)
        self.node_count = 0

    def _node(self, name):
        self.node_count = max(self.node_count, name + 1)
        return name

    def add_resistor(self, name, a, b, value):
        self._node(max(a, b))
        self.components.append(('R', name, a, b, value))

    def add_capacitor(self, name, a, b, value):
        self._node(max(a, b))
        self.components.append(('C', name, a, b, value))

    def add_inductor(self, name, a, b, value):
        self._node(max(a, b))
        self.components.append(('L', name, a, b, value))

    def add_vsource(self, name, a, b, value):
        self._node(max(a, b))
        self.components.append(('V', name, a, b, value))

    def add_isource(self, name, a, b, value):
        self._node(max(a, b))
        self.components.append(('I', name, a, b, value))


def _build_mna(circuit, omega=0.0):
    """构建 MNA 矩阵。omega=0 为 DC；omega>0 为 AC（复导纳）。"""
    n = circuit.node_count          # 节点 0 为地，实际未知节点 1..n-1
    # 电压源数量
    vsources = [c for c in circuit.components if c[0] == 'V']
    m = len(vsources)
    size = (n - 1) + m               # 去掉地节点，加电压源电流变量

    G = np.zeros((size, size), dtype=complex if omega > 0 else float)
    z = np.zeros(size, dtype=complex if omega > 0 else float)

    j = 1j if omega > 0 else 0

    def stamp_admittance(a, b, y):
        if a > 0:
            G[a - 1, a - 1] += y
        if b > 0:
            G[b - 1, b - 1] += y
        if a > 0 and b > 0:
            G[a - 1, b - 1] -= y
            G[b - 1, a - 1] -= y

    for kind, name, a, b, value in circuit.components:
        if kind == 'R':
            if value <= 0:
                value = 1e-12
            stamp_admittance(a, b, 1.0 / value)
        elif kind == 'C':
            y = j * omega * value if omega > 0 else 0.0  # DC 时电容开路
            if omega > 0:
                stamp_admittance(a, b, y)
        elif kind == 'L':
            if omega > 0:
                y = 1.0 / (j * omega * value)
                stamp_admittance(a, b, y)
            else:  # DC 时电感短路：直接用极小电阻近似
                stamp_admittance(a, b, 1e12)
        elif kind == 'I':
            if a > 0:
                z[a - 1] -= value
            if b > 0:
                z[b - 1] += value

    # 电压源：B 矩阵 + 电流变量
    for k, (kind, name, a, b, value) in enumerate(vsources):
        col = (n - 1) + k
        if a > 0:
            G[a - 1, col] += 1
            G[col, a - 1] += 1
        if b > 0:
            G[b - 1, col] -= 1
            G[col, b - 1] -= 1
        z[col] = value

    return G, z


def solve_dc(circuit):
    """DC 工作点分析，返回节点电压向量（含地节点 0）。"""
    G, z = _build_mna(circuit, omega=0.0)
    x = np.linalg.solve(G, z)
    n = circuit.node_count
    v = np.zeros(n)
    v[1:] = x[:n - 1]
    return v


def solve_ac(circuit, freq):
    """AC 频响分析，返回指定频率下的复数节点电压（含地节点 0）。"""
    omega = 2 * np.pi * freq
    G, z = _build_mna(circuit, omega=omega)
    x = np.linalg.solve(G, z)
    n = circuit.node_count
    v = np.zeros(n, dtype=complex)
    v[1:] = x[:n - 1]
    return v
