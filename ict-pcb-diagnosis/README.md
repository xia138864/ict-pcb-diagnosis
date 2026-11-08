# 少样本 PCB 故障诊断：知识图谱嵌入与门控融合

本目录是论文《基于知识图谱嵌入与门控融合的 PCB 少样本故障诊断方法》的**完整实验代码与结果**，
对应 22 类电路故障、k=5/10/20/50 少样本设置，以及基于属性的零样本诊断实验。

电路响应由改进节点分析（MNA）求解器按 SPICE 网表语义生成，**不依赖任何第三方版权数据**，
数据可由脚本完全重建。

---

## 1. 目录结构

```
ict-pcb-diagnosis/
├── spice_sim/                  # 电路仿真与数据集生成
│   ├── mna.py                  # MNA 直流/交流求解器（纯 Python）
│   └── generate_data_ext.py    # 22 类故障 × 100 维测量向量（8800 样本）
├── kg/                         # 知识图谱构建与嵌入
│   ├── clean_graph.py          # LLM 抽取图谱 → 规范图谱
│   ├── build_kg_22.py          # 22 类故障节点补全 + 关系规范化
│   ├── attribute_layer.py      # 故障→现象→表现 属性矩阵（8 属性）
│   ├── fault_lib_ext.py        # 22 类故障库定义
│   ├── transe.py               # TransE 预训练（d=128）
│   ├── kg_branch.py            # 1 跳子图池化（K_HOP=1）
│   ├── fault_graph_clean.json  # 图谱：1037 节点 / 963 边
│   ├── class_attributes.json   # 类别 × 属性 矩阵
│   ├── transe_embeddings.pt    # 预训练嵌入
│   └── prototypes_subgraphs.pt # 类别子图原型
├── src/
│   ├── dataset.py              # 数据加载与少样本采样
│   ├── model.py                # MLP / KGProtoNet / KGGuidedModel
│   ├── gating.py               # 多维门控融合
│   ├── train.py                # 训练、早停、分层划分
│   ├── run_experiments.py      # 主实验（3 模型 × 4 个 k × 5 seeds）
│   ├── ablation.py             # 消融实验（11 个变体）
│   └── zeroshot_attr_ext.py    # 属性零样本诊断（DAP）
├── data/
│   ├── spice_dataset.npz       # 22 类仿真数据（8800×100）
│   └── spice_meta_ext.json     # 类别、属性、测量维度元信息
├── docs/                       # 各阶段实验报告
└── results_*.json              # 原始实验结果
```

## 2. 环境

```bash
python >= 3.10
pip install -r requirements.txt
```

GPU 可选；CPU 亦可完成全部实验（主实验约 3–4 小时，视机器而定）。
为消除 CUDA 非确定性，训练脚本已固定 `torch.backends.cudnn.deterministic = True`。

## 3. 快速验证（使用随附数据，约 1 分钟）

```bash
cd ict-pcb-diagnosis
python - <<'EOF'
import sys; sys.path[:0] = ['src', 'kg']
from dataset import load_data, CLASSES
X, y = load_data()
print(X.shape, len(CLASSES))          # (8800, 100) 22
EOF
```

单次训练：

```bash
python src/train.py --model kg2 --k 10 --seed 2024
# --model 可选 mlp / kgproto / kg / kg2，其中 kg2 为论文完整模型（Ours）
```

## 4. 复现全部实验

### 4.1 主实验（表 2）

```bash
python src/run_experiments.py \
    --k-list 5 10 20 50 \
    --seeds 2024 2025 2026 2027 2028 \
    --epochs 100
# 输出 results_spice.json
```

### 4.2 消融实验（表 3）

```bash
for k in 5 10 20 50; do
    python src/ablation.py --k $k --seeds 2024 2025 2026 2027 2028 \
        --out results_ablation_k$k.json
done
```

覆盖 11 个变体：A1 拼接 / A2 相加 / A3 固定门控 / C1 无知识分支 / C2 可学习原型 /
B1 硬检索 / B2 均匀知识 / D1 0 跳 / D2 2 跳 / E1 随机嵌入，以及完整模型 Ours。
所有变体共用同一随机协议（相同噪声序列与 batch 顺序），差异仅来自模型结构。

### 4.3 属性零样本诊断（表 4 / 图 5）

```bash
python src/zeroshot_attr_ext.py --k 20 --seeds 2024 2025 2026 --folds-per-level 6
# 输出 results_zeroshot_attr_ext.json
```

### 4.4 从零重建数据与图谱（可选）

随附数据已可直接使用；如需重建：

```bash
python spice_sim/generate_data_ext.py     # 生成 data/spice_dataset_ext.npz
cp data/spice_dataset_ext.npz data/spice_dataset.npz
python kg/build_kg_22.py                  # 图谱 + 属性矩阵
python kg/transe.py                       # TransE 嵌入
```

## 5. 主要结果

### 5.1 少样本诊断准确率（%，5 seeds，均值±标准差）

| k | MLP | KG-Proto（无门控） | Ours（门控融合） |
|---|---|---|---|
| 5  | 44.26±14.14 | 64.85±2.46 | **71.48±0.85** |
| 10 | 61.95±1.61  | 71.70±1.42 | **74.33±1.42** |
| 20 | 67.20±2.22  | 75.59±0.85 | **76.64±0.32** |
| 50 | 69.05±2.23  | 76.91±1.13 | **77.36±1.06** |

配对 t 检验：k=5 时 Ours 相对 MLP 提升 27.23 个百分点（p=0.013），
相对纯原型提升 6.64 个百分点（p=0.006）；k=10 时相对 MLP 提升 12.38 个百分点（p<0.001）。

### 5.2 属性零样本（3 类留出，k=20，10 折 min_cov≥3）

| 方法 | 准确率 |
|---|---|
| KG 属性矩阵（DAP） | **54.9±12.9%** |
| 随机属性矩阵 | 25.5±7.1% |

随机猜测为 33.3%（3 类）。10 折中 9 折优于随机对照，配对检验 p=0.0003。

## 6. 说明

- 消融显示：融合结构、知识分支的贡献显著；而知识**内容**的替换（硬检索/均匀/随机嵌入）
  在 22 类设定下差异不显著。论文已如实报告这一现象，并将其归因于
  「1 跳子图中故障节点与故障元件的强区分性 + 小样本下分类器主导拟合」。
- `data/spice_dataset.npz` 与 `spice_meta_ext.json` 由本仓库脚本生成，可无损重建。
- 原始 LLM 抽取图谱由本仓库上层 `run_data/ict_fault_graph.json` 提供（`kg/clean_graph.py` 读取）。
