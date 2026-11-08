# -*- coding: utf-8 -*-
"""
生成论文插图（v4：数据全部从实验结果文件读取，避免手工誊写错误）
================================================================
- 图1 架构图
- 图2 少样本准确率对比（MLP / KG 原型 / Ours）
- 图3 增益随样本规模变化
- 图4 消融实验：(a) 融合机制消融  (b) k=10 全部消融变体

数据源：
  ict-pcb-diagnosis/results_spice.json          （主实验）
  ict-pcb-diagnosis/results_ablation_k{5,10,20,50}.json（消融）
"""
import sys
import json

sys.path.insert(0, 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/pylibs')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT_DIR = 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/'
RES = 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/ict-pcb-diagnosis/'

k_list = [5, 10, 20, 50]

# ---------------- 读取主实验结果 ----------------
_main = json.load(open(RES + 'results_spice.json', encoding='utf-8'))['results']
mlp = [np.mean(_main[f'mlp_k{k}']) * 100 for k in k_list]
mlp_std = [np.std(_main[f'mlp_k{k}']) * 100 for k in k_list]
kgproto = [np.mean(_main[f'kgproto_k{k}']) * 100 for k in k_list]
kgproto_std = [np.std(_main[f'kgproto_k{k}']) * 100 for k in k_list]
ours = [np.mean(_main[f'kg2_k{k}']) * 100 for k in k_list]
ours_std = [np.std(_main[f'kg2_k{k}']) * 100 for k in k_list]

# ---------------- 读取消融结果 ----------------
_abl = {k: json.load(open(RES + f'results_ablation_k{k}.json', encoding='utf-8'))['results'] for k in k_list}


def abl(name, k, stat='mean'):
    a = np.array(_abl[k][name]) * 100
    return a.mean() if stat == 'mean' else a.std()


# ---------------- 配色 ----------------
C_MLP = '#b0bec5'
C_PROTO = '#81d4fa'
C_OURS = '#1565c0'
C_A1 = '#ffb74d'
C_A2 = '#a5d6a7'
C_A3 = '#ef9a9a'


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, title, sub='', fc='#fafafa', ec='#546e7a'):
        b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                           fc=fc, ec=ec, linewidth=1.4, alpha=0.95)
        ax.add_patch(b)
        if sub:
            ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center',
                    fontsize=11.5, fontweight='bold', color='#263238')
            ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center',
                    fontsize=9, color='#607d8b')
        else:
            ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                    fontsize=11.5, fontweight='bold', color='#263238')

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18,
                            linewidth=1.6, color='#37474f')
        ax.add_patch(a)

    box(0.3, 4.6, 2.0, 1.0, '输入', '100 点测量向量', fc='#eceff1', ec='#78909c')
    box(3.0, 4.6, 2.6, 1.0, '数据分支', 'MLP 编码器', fc='#e3f2fd', ec='#1976d2')
    box(3.0, 2.3, 2.6, 1.0, '知识分支', 'TransE + 子图池化', fc='#fff3e0', ec='#f57c00')
    box(3.0, 0.7, 2.6, 0.9, '知识图谱', '1006 实体 · 877 关系', fc='#fce4ec', ec='#c2185b')
    box(6.4, 3.1, 2.5, 1.7, '多维门控融合', 'h_f = α⊙h_d + (1−α)⊙h_k', fc='#e8f5e9', ec='#2e7d32')
    box(9.6, 3.45, 2.0, 1.0, '分类器', '8 类故障', fc='#ede7f6', ec='#5e35b1')

    arrow(2.3, 5.1, 3.0, 5.1)
    arrow(5.6, 5.1, 6.4, 4.55)
    arrow(5.6, 2.8, 6.4, 3.45)
    arrow(8.9, 3.95, 9.6, 3.95)
    arrow(4.3, 2.3, 4.3, 1.6)
    arrow(4.3, 4.6, 4.3, 3.25)

    ax.text(5.8, 5.4, 'h_d', fontsize=10, color='#1976d2', style='italic')
    ax.text(5.8, 2.65, 'h_k', fontsize=10, color='#f57c00', style='italic')
    fig.tight_layout()
    fig.savefig(OUT_DIR + 'fig1_architecture.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('图1 完成')


def fig_comparison():
    x = np.arange(len(k_list))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    bars1 = ax.bar(x - w, mlp, w, yerr=mlp_std, label='MLP 基线', color=C_MLP,
                   edgecolor='#78909c', linewidth=0.8, capsize=3, error_kw=dict(lw=0.9, ecolor='#90a4ae'))
    bars2 = ax.bar(x, kgproto, w, yerr=kgproto_std, label='KG 原型', color=C_PROTO,
                   edgecolor='#4fc3f7', linewidth=0.8, capsize=3, error_kw=dict(lw=0.9, ecolor='#90a4ae'))
    bars3 = ax.bar(x + w, ours, w, yerr=ours_std, label='Ours (门控融合)', color=C_OURS,
                   edgecolor='#0d47a1', linewidth=0.8, capsize=3, error_kw=dict(lw=0.9, ecolor='#90a4ae'))

    for bars, vals in [(bars1, mlp), (bars2, kgproto), (bars3, ours)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8, color='#37474f')

    ax.set_xticks(x)
    ax.set_xticklabels([f'k = {k}' for k in k_list], fontsize=11)
    ax.set_ylabel('诊断准确率 (%)', fontsize=11)
    ax.set_xlabel('每类标注样本数 k', fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR + 'fig2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('图2 完成')


def fig_gain():
    ours_gain = [o - m for o, m in zip(ours, mlp)]
    proto_gain = [g - m for g, m in zip(kgproto, mlp)]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.fill_between(k_list, ours_gain, 0, color=C_OURS, alpha=0.12, zorder=1)
    ax.plot(k_list, ours_gain, 'o-', color=C_OURS, linewidth=2.6, markersize=10,
            markerfacecolor='white', markeredgewidth=2.2, label='Ours（门控融合）', zorder=3)
    ax.plot(k_list, proto_gain, 's--', color='#90a4ae', linewidth=1.8, markersize=7,
            markerfacecolor='white', markeredgewidth=1.5, label='KG 原型', zorder=2)
    ax.axhline(0, color='#9e9e9e', linewidth=1, linestyle=':')

    for k, g in zip(k_list, ours_gain):
        ax.annotate(f'+{g:.1f}', (k, g), textcoords='offset points', xytext=(12, -14),
                    ha='center', fontsize=10.5, color=C_OURS, fontweight='bold')

    ax.set_xlabel('每类标注样本数 k', fontsize=11)
    ax.set_ylabel('相对 MLP 基线的增益（百分点）', fontsize=11)
    ax.set_xticks(k_list)
    ax.set_xticklabels([f'{k}' for k in k_list])
    ax.set_xlim(2, 53)
    ax.set_ylim(-5, max(ours_gain) + 12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR + 'fig3_gain.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('图3 完成')


def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    # ---- (a) 融合机制消融 ----
    ax = axes[0]
    names_a = ['Ours (完整模型)', 'A1 拼接融合', 'A2 相加融合', 'A3 固定门控 α=0.5']
    labels_a = ['Ours（门控）', '拼接融合', '相加融合', '固定门控 α=0.5']
    colors_a = [C_OURS, C_A1, C_A2, C_A3]
    x = np.arange(len(k_list))
    w = 0.2
    for i, (nm, lb, c) in enumerate(zip(names_a, labels_a, colors_a)):
        vals = [abl(nm, k) for k in k_list]
        stds = [abl(nm, k, 'std') for k in k_list]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=stds, label=lb, color=c,
               edgecolor='#546e7a', linewidth=0.6, capsize=2,
               error_kw=dict(lw=0.8, ecolor='#90a4ae'))
    ax.set_xticks(x)
    ax.set_xticklabels([f'k = {k}' for k in k_list], fontsize=11)
    ax.set_ylabel('诊断准确率 (%)', fontsize=11)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4,
              frameon=False, fontsize=9.5)
    ax.set_title('(a) 融合机制消融', fontsize=12, pad=10)

    # ---- (b) k=10 全部消融变体（横向条形）----
    ax2 = axes[1]
    kk = 10
    order = ['C1 无知识分支(纯MLP)', 'A3 固定门控 α=0.5', 'A2 相加融合', 'A1 拼接融合',
             'C2 可学习类别原型(无KG)', 'B1 硬检索知识', 'E1 无预训练(随机嵌入)',
             'D1 0 跳(仅故障节点)', 'D2 2 跳子图', 'B2 均匀知识信号', 'Ours (完整模型)']
    short = ['无知识分支', '固定门控 α=0.5', '相加融合', '拼接融合',
             '可学习原型(无KG)', '硬检索知识', '随机嵌入(无预训练)',
             '0 跳子图', '2 跳子图', '均匀知识信号', 'Ours（完整模型）']
    vals = [abl(nm, kk) for nm in order]
    stds = [abl(nm, kk, 'std') for nm in order]
    cols = ['#b0bec5'] * (len(order) - 1) + [C_OURS]
    y = np.arange(len(order))
    ax2.barh(y, vals, xerr=stds, color=cols, edgecolor='#546e7a', linewidth=0.6,
             capsize=2, error_kw=dict(lw=0.8, ecolor='#90a4ae'))
    ax2.axvline(ours[1], color=C_OURS, linestyle='--', linewidth=1.4, alpha=0.75)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short, fontsize=9.5)
    ax2.set_xlabel('诊断准确率 (%)', fontsize=11)
    ax2.set_xlim(0, 112)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.35)
    ax2.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax2.spines[sp].set_visible(False)
    ax2.set_title('(b) k = 10 各消融变体', fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR + 'fig4_ablation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('图4 完成')


if __name__ == '__main__':
    fig_architecture()
    fig_comparison()
    fig_gain()
    fig_ablation()
    print('全部完成')
