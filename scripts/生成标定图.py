# -*- coding: utf-8 -*-
"""生成属性覆盖度标定曲线（图 5）"""
import sys
sys.path.insert(0, 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/pylibs')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/'
RES = OUT + 'ict-pcb-diagnosis/'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 8 类库（min_cov ≤ 1）
sweep = json.load(open(RES + 'results_coverage_sweep.json', encoding='utf-8'))
c8 = {c['min_cov']: c['acc_mean'] * 100 for c in sweep['curve']}
# 22 类库（min_cov 2~3）
ext = json.load(open(RES + 'results_zeroshot_attr_ext.json', encoding='utf-8'))
c22 = {c['min_cov']: (c['kg_matrix'], c['random_matrix']) for c in ext['curve']}

xs = [0, 1, 2, 3]
kg = [c8.get(0), c8.get(1), c22.get(2, (None,))[0], c22.get(3, (None,))[0]]
rnd = [None, None, c22.get(2, (None, None))[1], c22.get(3, (None, None))[1]]
chance = 100 / 3

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.axhline(chance, color='#9e9e9e', linestyle=':', linewidth=1.4, zorder=1)
ax.text(3.05, chance + 1.2, '随机猜测 33.3%', fontsize=9, color='#757575', va='bottom', ha='right')

ax.plot(xs, kg, 'o-', color='#1565c0', linewidth=2.6, markersize=10,
        markerfacecolor='white', markeredgewidth=2.2, label='知识图谱属性矩阵', zorder=3)
xs_r = [x for x, v in zip(xs, rnd) if v is not None]
rnd_r = [v for v in rnd if v is not None]
ax.plot(xs_r, rnd_r, 's--', color='#90a4ae', linewidth=1.8, markersize=8,
        markerfacecolor='white', markeredgewidth=1.6, label='随机属性矩阵', zorder=2)

for x, v in zip(xs, kg):
    if v is not None:
        ax.annotate(f'{v:.1f}', (x, v), textcoords='offset points', xytext=(0, 11),
                    ha='center', fontsize=10, color='#1565c0', fontweight='bold')

# 分区标注
ax.axvspan(-0.25, 1.25, color='#ef5350', alpha=0.07, zorder=0)
ax.axvspan(1.75, 3.25, color='#43a047', alpha=0.08, zorder=0)
ax.text(0.5, 62, '8 类库\n（min_cov ≤ 1）', ha='center', fontsize=9.5, color='#c62828')
ax.text(2.5, 62, '22 类库\n（min_cov 2~3）', ha='center', fontsize=9.5, color='#2e7d32')

ax.set_xlabel('属性覆盖度 min_cov（各属性中「可见集正例类别数」的最小值）', fontsize=10.5)
ax.set_ylabel('零样本诊断准确率 (%)', fontsize=11)
ax.set_xticks(xs)
ax.set_xticklabels(['0', '1', '2', '3'])
ax.set_xlim(-0.3, 3.4)
ax.set_ylim(0, 72)
ax.yaxis.grid(True, linestyle='--', alpha=0.35)
ax.set_axisbelow(True)
for sp in ['top', 'right', 'left']:
    ax.spines[sp].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig(OUT + 'fig5_coverage.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('图5 完成:', xs, kg, rnd)
