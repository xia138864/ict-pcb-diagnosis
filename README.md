# 论文材料总览

本目录归档论文《基于知识图谱嵌入与门控融合的 PCB 少样本故障诊断方法》的全部可复现材料。

```
paper/
├── ict-pcb-diagnosis/          # 实验代码、知识图谱、仿真数据、结果 JSON
│   ├── README.md               # 环境、复现命令、结果表
│   ├── spice_sim/              # MNA 电路求解器 + 22 类故障数据生成
│   ├── kg/                     # 图谱（1037 节点/963 边）、属性矩阵、TransE
│   ├── src/                    # 模型、主实验、消融、零样本
│   ├── data/                   # spice_dataset.npz (8800×100)
│   └── docs/                   # 分阶段实验报告
├── manuscript/
│   ├── 论文_修订版.docx         # 投稿正文（含 5 图 3 表）
│   ├── 论文章节_摘要引言.md      # 正文章节（Markdown 版，与 Word 一致）
│   ├── 论文章节_方法与实验.md
│   ├── figs/fig1–fig5.png      # 插图源文件
│   └── *实验报告.md             # 各实验的过程记录
└── scripts/
    ├── 生成论文word.py          # 从结果 JSON 生成 Word（数字不硬编码）
    ├── 生成插图.py              # 图 1–4
    ├── 生成标定图.py            # 图 5（属性覆盖度曲线）
    └── 导出章节md.py            # 由 Word 反向导出章节 md，避免两版不一致
```

## 复现路径

1. 按 `ict-pcb-diagnosis/README.md` 安装依赖（`requirements.txt`）。
2. 直接使用随附 `data/spice_dataset.npz`，或运行 `spice_sim/generate_data_ext.py` 重建数据。
3. 主实验 → 消融 → 零样本，三条命令见代码目录 README。
4. 结果 JSON 落在 `ict-pcb-diagnosis/results_*.json`。
5. `scripts/生成论文word.py` 读取这些 JSON 生成 Word；`生成插图.py`、`生成标定图.py` 生成插图。

## 数据说明

电路响应由改进节点分析（MNA）求解器按 SPICE 网表语义生成，覆盖 22 类故障（开路、短路、容值漂移、元件缺失、虚焊、桥连、间歇），
每样本 100 维测量向量，共 8800 条。知识图谱由 LLM 抽取结果清洗、补全而来，
脚本 `kg/clean_graph.py` 读取仓库上层 `run_data/ict_fault_graph.json`。
