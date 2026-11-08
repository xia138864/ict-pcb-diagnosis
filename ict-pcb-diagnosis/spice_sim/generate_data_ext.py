# -*- coding: utf-8 -*-
"""
扩展故障库数据生成 + 可测性检查
================================
1. MNA 仿真生成 22 类故障 × 400 样本（沿用主实验电路与 SNR=20dB 噪声设定）；
2. 可测性检查：对每类计算「最大节点偏离 z」，确认该故障在测量空间中确实可辨识
   （用于剔除物理上不可测的故障，如因并联低阻元件而几乎无影响的虚焊）。

输出：
  data/spice_dataset_ext.npz
  data/spice_meta_ext.json
  kg/detectability_report.json
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'kg'))

from generate_data import (BASE_COMPONENTS, apply_fault, monte_carlo, measure,
                           TEST_NODES, AC_NODES, DIM)  # noqa: E402
from fault_lib_ext import (CLASS_NAMES_EXT, FAULT_SPECS_EXT, FAULT_PHEN_EXT,
                           class_attributes, ATTRIBUTES)  # noqa: E402

OUT_NPZ = os.path.join(ROOT, 'data', 'spice_dataset_ext.npz')
OUT_META = os.path.join(ROOT, 'data', 'spice_meta_ext.json')
OUT_REPORT = os.path.join(ROOT, 'kg', 'detectability_report.json')

N_PER_CLASS = 400


def node_view(feat):
    """返回 {节点: (DC 电压, AC 频谱 15 维)}"""
    dc = {n: feat[i] for i, n in enumerate(TEST_NODES)}
    ac = feat[7:97].reshape(len(AC_NODES), 15)
    a = {n: ac[i] for i, n in enumerate(AC_NODES)}
    return dc, a


def main():
    rng = np.random.default_rng(2024)

    # 健康基线
    b_dc, b_ac = [], []
    for _ in range(300):
        f = measure(monte_carlo(BASE_COMPONENTS, rng))
        dc, a = node_view(f)
        b_dc.append(dc); b_ac.append(a)
    dc_mu = {n: np.mean([x[n] for x in b_dc]) for n in TEST_NODES}
    dc_sd = {n: np.std([x[n] for x in b_dc]) + 1e-9 for n in TEST_NODES}
    ac_mu = {n: np.mean([x[n] for x in b_ac], axis=0) for n in AC_NODES}
    ac_sd = {n: np.std([x[n] for x in b_ac], axis=0) + 1e-9 for n in AC_NODES}

    X, y, report = [], [], {}
    for ci, cls in enumerate(CLASS_NAMES_EXT):
        spec = FAULT_SPECS_EXT[cls]
        dcs, acs = [], []
        for _ in range(N_PER_CLASS):
            f = measure(monte_carlo(apply_fault(BASE_COMPONENTS, spec, rng), rng))
            X.append(f); y.append(ci)
            dc, a = node_view(f)
            dcs.append(dc); acs.append(a)

        z_dc, z_ac = {}, {}
        for n in TEST_NODES:
            z_dc[n] = abs(np.mean([x[n] for x in dcs]) - dc_mu[n]) / dc_sd[n]
            if n in AC_NODES:
                m = np.mean([x[n] for x in acs], axis=0)
                z_ac[n] = float(np.max(np.abs(m - ac_mu[n]) / ac_sd[n]))
            else:
                z_ac[n] = 0.0
        report[cls] = {'z_dc': z_dc, 'z_ac': z_ac,
                       'max_z': float(max(max(z_dc.values()), max(z_ac.values()))),
                       'phenomenon': FAULT_PHEN_EXT[cls],
                       'attributes': [a for a, v in class_attributes(cls).items() if v]}

    X = np.array(X); y = np.array(y, dtype=np.int64)
    X = X + rng.normal(0.0, X.std(axis=0, keepdims=True) * 0.1, size=X.shape)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X, y=y)
    json.dump({'classes': CLASS_NAMES_EXT, 'dim': DIM, 'attributes': ATTRIBUTES,
               'phenomenon': FAULT_PHEN_EXT,
               'class_attributes': {c: class_attributes(c) for c in CLASS_NAMES_EXT}},
              open(OUT_META, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(report, open(OUT_REPORT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('=== 扩展数据集生成完成 ===')
    print(f'类别数: {len(CLASS_NAMES_EXT)}  样本数: {X.shape[0]}  维度: {X.shape[1]}')
    print(f'已保存: {OUT_NPZ}\n')
    print('=== 可测性检查（max_z 越大越易辨识；<6 视为物理上不可测）===')
    weak = []
    for cls in CLASS_NAMES_EXT:
        r = report[cls]
        flag = 'OK  ' if r['max_z'] >= 6 else 'WEAK'
        if r['max_z'] < 6:
            weak.append(cls)
        print(f'  [{flag}] {cls:<18} max_z={r["max_z"]:8.1f}')
    print(f'\n弱可测类别: {weak if weak else "无"}')


if __name__ == '__main__':
    main()
