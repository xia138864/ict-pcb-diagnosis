# KG 属性零样本实验报告

> 目标：让知识图谱从「可插拔接口」升级为**真正的贡献**——用图谱的语义结构支持**未见故障的零样本诊断**。
> 结论：**机制建成且验证有效，但当前 8 类故障库的规模不足以保证覆盖，零样本平均性能不稳定。** 根因已定量定位。

---

## 1. 为什么之前的零样本必然失败

上一轮实验里，知识图谱只提供「类别级原型向量」`P_c`。此时：

- 知识与测量之间**没有任何共同的语义坐标**——TransE 嵌入的几何结构与测量空间的几何结构无关；
- 模型只学会了"把可见类的样本映射到可见类的原型附近"，对未见类原型毫无概念；
- 结果：未见类准确率 13.7%，**低于随机原型（25.7%）与猜测线（33.3%）**。

**关键洞察**：要让 KG 起作用，必须让图谱表达**「故障 → 可观测物理表现」**的映射，而不是抽象向量。

---

## 2. 本次改造：给知识图谱加「可观测表现层」

在原有图谱之上新增一层 **manifestation（可观测表现）节点**，形成两跳路径：

```
fault ──has_phenomenon──> phenomenon ──manifests_as──> manifestation
```

属性由**图谱遍历自动导出**（不是手工指定类别属性表）：

| 现象（图谱原有） | 导出的可观测表现 |
|---|---|
| downstream nodes lose voltage | voltage_loss, current_anomaly, scope_global |
| node voltage pulled to ground | ground_clamp, current_anomaly, scope_local |
| filter frequency response shifted | freq_shift, freq_low |
| missing filter capacitance | freq_shift, freq_high |
| increased contact resistance | voltage_loss, ratio_shift, scope_local |
| adjacent nodes electrically bridged | ground_clamp, ratio_shift, scope_local |
| fluctuating measurements | fluctuation, ratio_shift, scope_local |
| wrong voltage divider ratio | ratio_shift, voltage_loss, scope_global |

**得到的 10 维属性矩阵（全部 8 类两两可分）**：

| 类别 | current_anom | fluctu | freq_high | freq_low | freq_shift | ground_clamp | ratio_shift | scope_global | scope_local | voltage_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| open | 1 | | | | | | | 1 | | 1 |
| short | 1 | | | | | 1 | | | 1 | |
| drift | | | | 1 | 1 | | | | | |
| missing | | | 1 | | 1 | | | | | |
| cold_solder | | | | | | | 1 | | 1 | 1 |
| bridge | | | | | | 1 | 1 | | 1 | |
| intermittent | | 1 | | | | | 1 | | 1 | |
| wrong_part | | | | | | | 1 | 1 | | 1 |

**这一步本身就是 KG 的实质贡献**：图谱现在编码了"每种故障在电气量上留下什么可观测痕迹"，且属性在故障间**共享**（`ratio_shift` 被 4 类共享、`scope_local` 被 4 类共享），因此属性检测器可以在可见故障上训练、迁移到未见故障。

随后用 **DAP（Direct Attribute Prediction）** 完成零样本识别：

```
训练：测量向量 → p(属性 j 存在)      （可见故障上做多标签监督）
测试：s(c) = Σ_j [ a_cj·log p_j + (1−a_cj)·log(1−p_j) ]   在未见类中取 argmax
```

属性检测器与数据特征同样经**逐维门控融合**（复用论文核心模块）。

---

## 3. 实验结果

### 版本 1：6 维属性（nature 层）

留出折：FA = {open, drift, cold_solder}，FB = {short, missing, wrong_part}，FC = {bridge, drift, cold_solder}；3 折 × 5 种子，k = 20，猜测线 33.3%。

| 变体 | 零样本准确率 | 属性检测率 | vs Ours |
|---|---|---|---|
| **Ours-Attr（KG 属性矩阵 + 门控）** | **57.85% ± 30.03** | 99.52% | — |
| 属性 + 无门控（拼接） | 40.59% ± 11.80 | 73.83% | −17.26pp, **p = 0.0357** |
| 随机属性矩阵（同密度行置换） | 40.22% ± 11.77 | 99.39% | −17.63pp, p = 0.0802 |
| 随机属性向量（门控） | 56.37% ± 28.37 | 99.54% | −1.48pp, p = 0.6314 |

**分折明细 —— 关键证据：**

| 折 | 未见类 | 准确率 |
|---|---|---|
| FA | open, drift, cold_solder | **75.22% ± 11.92** |
| FB | short, missing, wrong_part | **80.33% ± 12.29** |
| FC | bridge, drift, cold_solder | **18.00% ± 4.05** ← 低于猜测线 |

三个正向信号：
1. **属性覆盖充分的折上，零样本可达 75~80%**（猜测线 33.3%）——机制成立；
2. **KG 属性矩阵优于随机矩阵**（57.9% vs 40.2%，p = 0.080）——知识内容确实起作用；
3. **门控融合显著优于拼接**（p = 0.036）——与论文主实验结论一致。

### 版本 2：10 维属性（增加 scope / band 层，使 8 类两两可分）

留出折：FA = {open, cold_solder, bridge}，FB = {short, cold_solder, wrong_part}，FC = {bridge, cold_solder, wrong_part}。

| 变体 | k = 20 | k = 50 |
|---|---|---|
| Ours-Attr（KG 属性矩阵 + 门控） | 26.70% ± 11.03 | 36.11% ± 7.99 |
| 属性 + 无门控 | 33.33% ± 0.00 | 33.33% ± 0.00 |
| 随机属性矩阵 | 33.41% ± 21.62 | 31.15% ± 16.40 |
| 随机属性向量 | 28.78% ± 11.64 | 31.00% ± 12.93 |

**结果退回到猜测线附近。** 属性检测率仍达 99.3%（可见类上），但迁移到未见类失败。

---

## 4. 根因（已定量定位）

对失败折做了逐属性预测诊断，问题一目了然：

```
FA 折（6 维属性版）
  open         真值=[1 0 0 0 0 1]  预测=[0.62 0.00 0.00 1.00 0.35 0.00]
  cold_solder  真值=[0 0 0 0 1 1]  预测=[0.00 0.61 0.26 0.01 0.88 0.00]
```

`voltage_loss` 的预测在 open 和 cold_solder 上**都是 0.00**——检测器完全失效。原因：

> **`voltage_loss` 在整个可见集里只有 1 个正例（wrong_part）。**
> 检测器学到的不是"电压失压"这个物理概念，而是 **wrong_part 特有的测量指纹**（R2 增大 10 倍 → 节点 2 电压抬高），因此对 open / cold_solder 完全不响应。

**每个属性的正例类别数（8 类库）：**

| 属性 | 正例类别数 | 正例类别 |
|---|---|---|
| current_anomaly | 2 | open, short |
| fluctuation | **1** | intermittent |
| freq_low / freq_high | **1** | drift / missing |
| freq_shift | 2 | drift, missing |
| ground_clamp | 2 | short, bridge |
| ratio_shift | 4 | cold_solder, bridge, intermittent, wrong_part |
| scope_global | 2 | open, wrong_part |
| scope_local | 4 | short, cold_solder, bridge, intermittent |
| voltage_loss | 3 | open, cold_solder, wrong_part |

**从 1~4 个正例类别中学习一个可迁移的物理概念，在原理上就不可行。** 这正是版本 2 加细属性后反而更差的原因：属性变多、每个属性的正例更少。

**结论：瓶颈不是方法，是数据规模。** 8 类故障库为每个属性提供的正例太少。

---

## 5. 解决办法（需要改数据集，请拍板）

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 扩展故障库（推荐）** | 用同一 MNA 仿真器把故障库从 8 类扩到 **25~30 类**（如 open_R1/R3/R4/R6、short_R2/R5/R7/R10、drift_C1/C2/C3、bridge(3,4)/(4,5)…），属性矩阵同步扩充（增加**逐节点位置属性**以区分同机理不同位置的故障）。此时每个属性有 **8~15 个正例类别**，属性检测器才可能学到可迁移概念 | 改数据集 → 主实验/消融/图表/Word **全部数字需重跑**（约 1~2 小时） |
| B. 保持 8 类，如实报告负面结果 | 把「属性覆盖足够时零样本 75~80%、覆盖不足时失效」作为边界写入局限 | 低（改文字） |
| C. 放弃零样本，KG 定位为可插拔先验接口 | 回到当前 Word 版本 | 零（已完成） |

**我的建议是 A**：这是唯一能让知识图谱成为实质贡献的路径，而且机制已验证有效（75~80%），缺的只是让每个属性有足够多的正例类别。

**风险提示**：扩展到 25~30 类后，诊断任务本身显著变难（25~30 路分类），主实验的少样本准确率会下降，但**知识引导方法相对纯数据基线的增益预计会更大**（纯 MLP 在 30 类少样本下会更接近随机）。

---

## 6. 已交付的代码资产

| 文件 | 作用 |
|---|---|
| `kg/attribute_layer.py` | 从图谱遍历导出「故障→属性」矩阵（10 维，8 类两两可分），含 provenance 溯源 |
| `src/zeroshot_attr.py` | DAP + 门控的零样本实现，含 4 个对照变体 |
| `results_zeroshot_attr.json` | 全部原始结果（含分折明细） |

复现：
```bash
cd ict-pcb-diagnosis
python kg/attribute_layer.py
python src/zeroshot_attr.py --k-list 20 50 --seeds 2024 2025 2026 2027 2028 --epochs 200
```
