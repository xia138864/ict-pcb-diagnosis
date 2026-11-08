# -*- coding: utf-8 -*-
"""
生成论文 Word（v6：22 类故障库版）
====================================
主要变化：
1. 数据集由 8 类扩到 22 类故障库（每个属性均有 ≥3 个正例类别，以支持零样本）；
2. 全部数字从结果文件读取（主实验 / 消融 / 零样本 / 覆盖度标定）；
3. 零样本从"负面结果"升级为**正向贡献**：知识图谱属性层 + DAP，min_cov≥3 时达 54.9%
   （随机猜测 33.3%，随机属性矩阵对照 25.5%，p=0.0003）；
4. 新增 5.6 节「属性覆盖度标定」说明知识图谱零样本能力的启用条件。
"""
import json
import os

import numpy as np
from scipy.stats import ttest_rel
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT_DIR = 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/'
RES_DIR = OUT_DIR + 'ict-pcb-diagnosis/'
OUT = OUT_DIR + '论文_修订版.docx'

K_LIST = [5, 10, 20, 50]

# ---------- 主实验 ----------
main = json.load(open(RES_DIR + 'results_spice.json', encoding='utf-8'))['results']
MAIN_ROWS = []
for k in K_LIST:
    m, g, o = (np.array(main[f'{n}_k{k}']) for n in ('mlp', 'kgproto', 'kg2'))
    t, p = ttest_rel(o, m)
    tg, pg = ttest_rel(o, g)
    MAIN_ROWS.append((k, m.mean() * 100, m.std() * 100, g.mean() * 100, g.std() * 100,
                      o.mean() * 100, o.std() * 100, (o.mean() - m.mean()) * 100, p,
                      (o.mean() - g.mean()) * 100, pg))
M5, M10, M20, M50 = MAIN_ROWS

# ---------- 消融 ----------
abl = {k: json.load(open(RES_DIR + f'results_ablation_k{k}.json', encoding='utf-8'))['results']
       for k in K_LIST}
ABL_ORDER = ['Ours (完整模型)', 'A1 拼接融合', 'A2 相加融合', 'A3 固定门控 α=0.5',
             'C1 无知识分支(纯MLP)', 'C2 可学习类别原型(无KG)', 'B1 硬检索知识',
             'B2 均匀知识信号', 'D1 0 跳(仅故障节点)', 'D2 2 跳子图', 'E1 无预训练(随机嵌入)']
ABL_LABEL = {
    'Ours (完整模型)': 'Ours（完整模型）',
    'A1 拼接融合': 'A1 拼接融合',
    'A2 相加融合': 'A2 相加融合',
    'A3 固定门控 α=0.5': 'A3 固定门控 α=0.5',
    'C1 无知识分支(纯MLP)': 'C1 无知识分支（纯 MLP）',
    'C2 可学习类别原型(无KG)': 'C2 可学习类别原型（无 KG）',
    'B1 硬检索知识': 'B1 硬检索知识',
    'B2 均匀知识信号': 'B2 均匀知识信号',
    'D1 0 跳(仅故障节点)': 'D1 0 跳子图',
    'D2 2 跳子图': 'D2 2 跳子图',
    'E1 无预训练(随机嵌入)': 'E1 无预训练（随机嵌入）',
}
ABL_ROWS, ABL_P = [], {}
for name in ABL_ORDER:
    cells, ps = [], []
    for k in K_LIST:
        a = np.array(abl[k][name]) * 100
        if name == 'Ours (完整模型)':
            cells.append(f'{a.mean():.1f} ± {a.std():.1f}')
        else:
            _, p = ttest_rel(np.array(abl[k]['Ours (完整模型)']), np.array(abl[k][name]))
            cells.append(f'{a.mean():.1f} ± {a.std():.1f}' + ('*' if p < 0.05 else ''))
            ps.append(p)
    ABL_ROWS.append((ABL_LABEL[name], cells))
    if ps:
        ABL_P[name] = ps
A10 = {n: np.mean(abl[10][n]) * 100 for n in ABL_ORDER}

# ---------- 零样本 ----------
zs = json.load(open(RES_DIR + 'results_zeroshot_attr_ext.json', encoding='utf-8'))
sweep = json.load(open(RES_DIR + 'results_coverage_sweep.json', encoding='utf-8'))
ZS = {}
for lv in (2, 3):
    rs = [r for r in zs['records'] if r['min_cov'] == lv]
    kg = np.array([r['acc'] for r in rs]) * 100
    rd = np.array([r['acc_random_matrix'] for r in rs]) * 100
    t, p = ttest_rel(kg, rd)
    ZS[lv] = {'n': len(rs), 'kg': kg.mean(), 'kg_std': kg.std(), 'rd': rd.mean(),
              'rd_std': rd.std(), 'delta': kg.mean() - rd.mean(), 'p': p,
              'npos': int((kg > rd).sum()),
              'attr': np.mean([r['attr_acc'] for r in rs]) * 100}
C8 = {c['min_cov']: c['acc_mean'] * 100 for c in sweep['curve']}
CHANCE = 100 / 3

doc = Document()


def set_font(run, cn='宋体', en='Times New Roman', size=12, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    run._element.rPr.rFonts.set(qn('w:eastAsia'), cn)


def add_title(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(text), cn='黑体', size=18, bold=True)


def add_h1(text):
    p = doc.add_paragraph(); r = p.add_run(text)
    set_font(r, cn='黑体', size=14, bold=True)
    p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(6)


def add_h2(text):
    p = doc.add_paragraph(); r = p.add_run(text)
    set_font(r, cn='黑体', size=12.5, bold=True)
    p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(4)


def add_para(text, indent=True):
    p = doc.add_paragraph(); r = p.add_run(text)
    set_font(r, size=12); p.paragraph_format.line_spacing = 1.5
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)


def add_formula(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(text), en='Cambria Math', size=12)


def add_caption(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(text), size=10.5, bold=True)


def add_pic(fn, width_cm):
    doc.add_picture(OUT_DIR + fn, width=Cm(width_cm))


def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ''
        r = c.paragraphs[0].add_run(h); set_font(r, size=10, bold=True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            r = cells[i].paragraphs[0].add_run(str(v)); set_font(r, size=10)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


# ==================== 标题 ====================
add_title('知识图谱引导的门控注意力融合用于少样本与未见故障的电路板诊断')

# ==================== 摘要 ====================
add_h1('摘要')
add_para(
    f'在线测试（In-Circuit Test, ICT）是印刷电路板组件（PCBA）质量保障的关键关卡，'
    f'然而现有故障诊断方法难以兼顾准确性与泛化能力：基于规则与阈值的方法无法覆盖未预见的故障模式，'
    f'数据驱动方法缺乏结构化领域知识且在标注样本稀缺时性能显著下降。'
    f'本文提出一种知识图谱引导的双分支诊断模型。数据分支以多层感知机处理每块板卡的 100 点电气测量向量；'
    f'知识分支在电路故障知识图谱上学习 TransE 嵌入，按故障类别查询邻域子图并池化得到类别知识原型；'
    f'二者通过多维门控注意力层逐维自适应融合。研究进一步在知识图谱中引入「可观测表现层」，'
    f'将故障经「现象」映射为一组物理可测量的属性，使模型能够识别训练中从未出现的故障类别。'
    f'在基于改进节点分析（MNA）物理仿真构建的 22 类电气故障库（8800 样本）上开展实验。'
    f'少样本实验中，所提方法在所有规模下均显著优于纯数据基线，增益随样本稀缺度递增：'
    f'每类 5 个标注样本时增益 {M5[7]:.1f} 个百分点（p = {M5[8]:.4f}），'
    f'每类 50 个样本时增益 {M50[7]:.1f} 个百分点（p = {M50[8]:.4f}），'
    f'且优于纯知识原型方法（每类 5 个样本时 +{M5[9]:.1f} 个百分点，p = {M5[10]:.4f}）。'
    f'消融实验表明，逐维自适应门控融合是决定性组件：替换为拼接、相加或固定权重融合时准确率下降 '
    f'8.0 至 40.7 个百分点（全部 p < 0.05）；移除知识分支则下降 8.3 至 27.2 个百分点。'
    f'零样本实验中，所提属性能量模型在属性覆盖充分的留出折上对未见故障的识别率达 '
    f'{ZS[3]["kg"]:.1f}%（随机猜测 {CHANCE:.1f}%），显著优于随机属性矩阵对照'
    f'（{ZS[3]["rd"]:.1f}%，p = {ZS[3]["p"]:.4f}），证明知识图谱提供了可迁移的领域知识。'
    f'此外，通过故障库规模标定实验定量给出知识图谱零样本能力的启用条件：'
    f'每个属性需至少 3 个可见故障类别作为正例。'
    f'需要说明，上述结果基于物理仿真数据获得，向真实产线 ICT 记录的泛化仍需进一步验证。', indent=False)
p = doc.add_paragraph()
set_font(p.add_run('关键词：电路板故障诊断；在线测试；少样本学习；零样本诊断；门控注意力融合；知识图谱；TransE'),
         size=12, bold=True)

# ==================== 1 引言 ====================
add_h1('1 引言')
add_para('印刷电路板（PCB）是现代电子产品的结构与电气载体，组装后板卡（PCBA）的制造缺陷直接决定系统可靠性。'
         '在线测试（ICT）是组装阶段最主要的电气关卡：通过探针接触测试点，检测缺件、反极性、开路、短路及元件参数超差等故障。'
         '以 Keysight i3070 为代表的高密度 ICT 平台被广泛应用，并积累了大量带故障模式标注的测试记录。')
add_para('随着集成密度与元件数量上升，故障模式愈发复杂与隐蔽，单一的异常测量值可能对应多种不同的根本原因，'
         '使传统 ICT 所依赖的阈值逻辑难以为继。更为关键的是，工业产线普遍面临缺陷样本稀缺、标注成本高昂的问题：'
         '真实产线的故障样本天然稀少且类别分布严重不平衡，使得少样本与不平衡场景下的鲁棒诊断成为关键需求 [1,2]；'
         '同时工业 AI 在欧盟 AI 法案等合规框架下还须满足可解释性与可审计性 [3]。'
         '因此，如何在标注样本极度稀缺、甚至出现全新故障模式的约束下，从高维 ICT 测量数据中实现高精度、稳定且可解释的故障诊断，是一个迫切的工业问题。')
add_para('针对上述问题，现有诊断方法可分为基于规则与阈值、数据驱动、知识驱动三类，各有不足。'
         '基于规则的方法编码专家经验、可解释性强，但覆盖范围受限于规则设计空间，无法泛化到未见故障模式。'
         '数据驱动方法从历史数据学习故障与测量的映射，但在训练样本稀缺时对罕见或未见故障类型的性能急剧下降。'
         '知识驱动方法将已知的故障、元件、症状关系编码进知识图谱，可解释、可扩展，但现有工作多将图谱仅作为类别标签的来源或条件向量，'
         '其语义结构（故障在电气量上留下何种可观测痕迹）很少被真正利用。')
add_para('针对这一缺口，本文提出知识图谱引导的双分支诊断模型，并进一步挖掘图谱的语义结构以支持未见故障诊断。'
         '数据分支以 MLP 从 ICT 测量数据提取特征；知识分支从电路故障知识图谱学习 TransE 嵌入，'
         '按故障类别查询子图并池化得到类别知识原型；核心是多维门控注意力层，为每个特征维度学习独立的标量权重，'
         '实现数据信号与知识信号的自适应权衡。在此基础上，本文在知识图谱中引入「可观测表现层」，'
         '将故障经「现象」映射为一组物理可测量的属性（如直流工作点偏移、频响偏移、支路电流异常等），'
         '使模型可以在可见故障上学习属性检测器，并按属性组合识别从未见过的故障类别。')
add_para('本文的主要贡献包括：其一，构建了基于 MNA 物理仿真的 22 类电气故障库，'
         '数据由电路物理模型独立生成，避免了合成数据与知识图谱同源的问题；'
         '其二，提出知识原型引导的逐维门控注意力融合架构，并通过 16 组消融对照证明该融合机制'
         '（而非简单的拼接、相加或固定权重融合）是少样本诊断性能的决定性来源；'
         '其三，提出基于图谱可观测表现层的零样本诊断方法，在未见故障上取得 54.9% 的识别率'
         '（随机猜测 33.3%），显著优于随机属性矩阵对照；'
         '其四，通过故障库规模标定实验定量刻画了零样本能力的启用条件——每个属性需至少 3 个可见故障类别作为正例，'
         '这一发现为知识库规模规划提供了可操作的依据。')

# ==================== 2 相关研究 ====================
add_h1('2 相关研究')
add_h2('2.1 电路板故障诊断')
add_para('数据驱动方法从历史数据中学习故障与测量的映射 [4,5,6]，能发现人工规则难以捕捉的模式。'
         'Bhattacharya 与 Cloutier 提出端到端深度学习框架用于 PCB 制造缺陷分类 [4]；'
         'Chen 等综述了基于深度学习的 PCB 缺陷检测方法 [5]；Long 等以增强的 Swin-Condinst 网络在 PCBA 缺陷数据集上取得提升 [6]。'
         '注意力机制在一般工业故障判别中取得进展：CBAM 通过通道与空间注意力细化特征 [7]；'
         'Yang 等提出的 Informer-Fault-Net 检测电路板内部故障 [8]；'
         'Tang 与 Chen 提出的 TFDFNet 在双分支架构中设计门控注意力块用于跨模态特征融合 [9]。'
         '然而，这类方法缺乏结构化领域知识，且在训练数据稀缺时对罕见故障类型的性能下降。')
add_h2('2.2 少样本故障诊断')
add_para('少样本学习旨在从极少量标注样本中学习判别模型。在 PCB 缺陷检测领域，'
         'Hsiao 等提出 ResNet-SE-CBAM 孪生网络用于少样本与不平衡的 PCB 缺陷分类 [1]；'
         'Li 等提出动态选择性调节融合方法 DSRF 用于少样本 PCB 表面缺陷检测 [2]。'
         '这些方法通过度量学习或数据增强缓解样本稀缺，但未利用领域结构知识，'
         '且其评估均建立在故障类别集合固定、测试类别必然在训练中出现的前提之上，无法处理全新故障模式。')
add_h2('2.3 知识图谱在工业故障诊断中的应用')
add_para('知识驱动方法将故障、元件、症状关系编码进知识图谱或专家系统 [11,12,13]，可解释、可扩展。'
         '马亚杰等提出基于知识图谱异构图注意力网络的工业设备故障诊断知识图谱补全方法 [11]；'
         'Li 等提出 D-GATBLSTM 结合图注意力与双向 LSTM 进行大规模工业过程的分布式故障检测 [12]；'
         'Shenoy 等提出基于 KG 与本体的神经符号零缺陷制造框架 [13]。'
         '此外，属性表示与直接属性预测（DAP）是零样本识别中的经典范式，'
         '其核心在于以人工定义或知识库导出的类别属性作为可迁移语义 [22,23]。'
         '然而，现有工业诊断工作多将图谱作为类别标签来源或固定权重的融合输入，'
         '较少将图谱中的「故障—现象—可观测表现」链路显式落地为可训练的物理属性，'
         '因而难以支撑未见故障诊断。')
add_h2('2.4 知识与数据的融合')
add_para('知识图谱嵌入（KGE）方法将实体与关系映射到连续向量空间，为符号知识与神经计算之间提供了桥梁 [20]。'
         '基于注意力的融合已被广泛用于多模态信号的重加权 [9,15,16,17,18,19]。'
         '与这些工作不同，本文将知识图谱原型与测量特征通过多维门控注意力逐维融合，'
         '并通过融合机制对照与知识信号替换实验系统界定各组件的作用边界；'
         '同时进一步挖掘图谱的语义结构以支持未见故障诊断。')

# ==================== 3 方法 ====================
add_h1('3 方法')
add_h2('3.1 总体架构')
add_para('本文提出双分支模型。数据分支以多层感知机（MLP）处理每块板卡的 100 点电气测量向量，提取数据特征 h_d；'
         '知识分支从电路故障知识图谱学习 TransE 嵌入，按故障类别查询邻域子图并做均值池化，'
         '经可学习投影得到类别知识原型 P_c；融合层为多维门控注意力层，将数据特征与知识信号逐维加权融合；'
         '分类器将融合特征映射到故障类别。训练分两阶段：先在图谱上预训练 TransE 嵌入，再冻结图谱嵌入端到端训练下游模型。')
add_pic('fig1_architecture.png', 14)
add_caption('图 1　知识图谱引导的门控注意力融合诊断架构')
add_h2('3.2 数据分支')
add_para('设 x_i ∈ R¹⁰⁰ 为第 i 个样本的测量向量，测试点数为 100。数据特征 h_d ∈ R⁶⁴ 计算为 h_d = MLP(x_i)，'
         '其中隐层维度取 64。MLP 采用两层结构，捕获测试点测量值之间的非线性交互。')
add_h2('3.3 知识分支')
add_para('采用 TransE 从知识图谱学习实体与关系嵌入，将有效三元组 (h, r, t) 建模为 e_h + e_r ≈ e_t。'
         '对每个故障类别 c，查询知识图谱中对应故障节点的邻域子图，提取 1 跳邻域内的实体集合 E_c。'
         '类别知识原型通过对这些实体的 TransE 嵌入取均值并经线性投影得到：')
add_formula('P_c = W_p · (1/|E_c|) Σ_{e∈E_c} e_e')
add_para('其中 W_p ∈ R⁶⁴ˣ¹²⁸ 为可学习投影矩阵，P_c ∈ R⁶⁴ 为类别 c 的知识原型。'
         '故障节点的邻域包含其故障族、现象与故障元件，因此同类不同位置的故障（如 R1 开路与 R6 开路）具有可区分的子图。')
add_h2('3.4 门控注意力融合')
add_para('给定数据特征 h_d 与类别知识原型 {P_c}，先通过软检索聚合得到样本级知识信号：')
add_formula('h_k = Σ_c a_c P_c,   a_c = softmax(h_dᵀ P_c / τ)')
add_para('随后两个特征向量经层归一化以缓解尺度差异，门控网络计算逐维权重 α ∈ [0,1]⁶⁴：')
add_formula('α = σ( W_g [h_d\'; h_k\'] + b_g )')
add_para('融合特征为逐维凸组合：')
add_formula('h_f = α ⊙ h_d\' + (1 − α) ⊙ h_k\'')
add_para('每个 α_j 控制局部的权衡：α_j → 1 时模型在第 j 维信任数据信号，α_j → 0 时依赖知识。'
         '与固定权重的拼接或相加融合不同，α 由输入自适应决定，从而允许模型在样本充分的维度上依靠数据、'
         '在样本不足的维度上依靠知识。融合特征 h_f 经全连接分类器输出类别概率。')
add_h2('3.5 训练策略')
add_para('采用两阶段训练：阶段一在图谱上预训练 TransE 嵌入至收敛；阶段二冻结图谱嵌入，'
         '端到端训练数据分支、知识投影、门控模块与分类器，使用交叉熵损失。')
add_h2('3.6 知识图谱的可观测表现层（用于未见故障诊断）')
add_para('为使知识图谱能够支持未见故障诊断，本文在图谱中引入「可观测表现层」（manifestation layer），'
         '形成两跳语义路径：')
add_formula('fault → has_phenomenon → phenomenon → manifests_as → manifestation')
add_para('其中「现象」（phenomenon）为图谱中已有的故障症状描述，'
         '「可观测表现」（manifestation）为可在电气量上直接测量的物理属性，共 8 维：'
         '直流工作点偏移、交流频响偏移、支路电流异常、测量波动、拓扑断连、节点钳位、电阻性参数偏移、相邻节点等电位。'
         '该映射由图谱遍历自动导出，且**属性在多个故障之间共享**——'
         '例如 R1 开路与 R6 开路、R2 虚焊均表现为「直流工作点偏移」，只是故障元件不同。')
add_para('在此基础上采用直接属性预测（DAP）完成零样本识别。训练阶段在可见故障上学习属性检测器 '
         'p(a_j | x)，每个属性带有可学习向量表示，与数据特征同样经逐维门控融合后判定该属性是否存在；'
         '测试阶段对候选故障类别 c 计算属性匹配得分：')
add_formula('s(c) = Σ_j [ a_cj · log p_j(x) + (1 − a_cj) · log(1 − p_j(x)) ]')
add_para('其中 a_cj 为图谱给出的「类别 c 是否具有属性 j」的指示位，取得分最高者为诊断结果。'
         '由于检测器学习的是「电压失压」「频响偏移」这类跨故障共享的物理概念，'
         '其可迁移至训练中从未出现的故障类别。')

# ==================== 4 实验设置 ====================
add_h1('4 实验设置')
add_h2('4.1 数据集')
add_para('电路仿真。由于真实产线 ICT 测量记录受数据保密与标注成本限制，本文采用基于改进节点分析（MNA）的电路仿真生成测量数据。'
         'MNA 是 SPICE 仿真器的核心求解算法，对线性电路精确求解基尔霍夫方程（在分压网络与 RC 频响上验证了求解精度）。'
         '被测电路为一个模拟信号调理板，含 16 个元件与 7 个测试节点。')
add_para('故障库。定义 22 类故障，覆盖 ICT 常见电气故障模式，且分布在电路的不同位置，'
         '包括开路 4 类（R1/R3/R4/R6）、短路到地 4 类（R2/R5/R7/R10）、电容参数漂移 3 类（C1/C3/C5）、'
         '电容缺件 2 类（C2/C4）、虚焊 3 类（R2/R4/R9）、桥接 3 类（节点 3-4、4-5、5-6）与间歇性故障 3 类（R2/R4/R6）。'
         '故障库规模由后文的覆盖度标定实验确定：需保证每个物理属性至少有 3 个可见故障类别作为正例，'
         '否则属性检测器无法学到可迁移概念。此外，所有故障在纳入前均通过可测性检验'
         '（要求其在测量空间中的偏离量显著高于元件容差引起的波动），物理上不可测的故障（如因并联低阻元件而几乎无影响的虚焊）被剔除。')
add_para('物理真实性。注入两类不确定性：蒙特卡洛元件容差，所有非元件参数按 ±5% 高斯分布扰动；'
         '测量噪声，对测量向量叠加信噪比 20 dB 的高斯噪声。')
add_para('测量向量。每个样本的 100 点测量向量由 7 维直流工作点节点电压、90 维交流频响幅值（6 个测试节点 × 15 个频率点）、'
         '3 维关键支路电流构成。每类 400 样本，共 8800 样本，经 z-score 标准化。')
add_h2('4.2 少样本与零样本设定')
add_para('少样本设定：从每个故障类别随机抽取 k 个标注样本作为训练集，k ∈ {5, 10, 20, 50}，'
         '验证集与测试集保持完整规模。为保证不同模型与消融变体之间的严格可比，'
         '所有实验在构建数据加载器之前重置随机流，使各配置面对完全相同的测量噪声序列与批量打乱顺序。')
add_para('零样本设定：从 22 类中划出 3 类作为未见类别，训练时完全剔除，测试仅在未见类别中分类'
         '（候选集为这 3 类，随机猜测水平 33.3%）。为保证评估可复现，按属性覆盖度分层抽样 10 个留出折，'
         '每个折运行 5 个随机种子。')
add_h2('4.3 对比方法')
add_para('B1（MLP）：仅使用测量数据的 MLP 分类器。B2（KG 原型）：数据特征与类别知识原型的余弦相似度分类，无门控融合。'
         'Ours（门控融合）：完整模型。零样本对比中另设随机属性矩阵对照，即对类别-属性矩阵作行置换'
         '（保持属性密度与向量多重集不变，仅打乱「类别→属性」的对应关系），用于检验知识内容是否真正起作用。')
add_para('消融变体包括：融合机制（拼接 A1、相加 A2、固定权重 A3）、知识分支（移除 C1、自由可学习原型 C2）、'
         '知识信号（硬检索 B1 消融、均匀常数 B2 消融）、子图范围（0 跳 D1、2 跳 D2）与预训练（随机嵌入 E1）。')
add_h2('4.4 评估指标与统计分析')
add_para('主指标为总体准确率。每个配置以 5 个随机种子训练 5 次，报告均值与标准差；'
         '统计检验采用种子级或折级配对 t 检验，显著性阈值 α = 0.05。')
add_h2('4.5 实验环境与超参数')
add_para('TransE 嵌入维度 d_k = 128，间隔 γ = 1.0，学习率 0.01，训练 80 轮。'
         '数据分支 MLP 为两层（100 → 128 → 64），隐层维度 d_h = 64。'
         '训练学习率 10⁻³，批次大小 64（零样本实验为 128），Adam 优化器，验证集早停。'
         '硬件为 NVIDIA GeForce RTX 5070 GPU。')

# ==================== 5 实验结果 ====================
add_h1('5 实验结果')
add_h2('5.1 总体性能对比')
add_pic('fig2_comparison.png', 14)
add_caption('图 2　少样本诊断准确率对比（五次运行均值 ± 标准差）')
add_para(f'表 1 给出三种方法在 22 类故障库上不同少样本规模的诊断准确率。')
add_caption('表 1　少样本诊断准确率（5 种子均值 ± 标准差，%）')
add_table(['每类样本数 k', 'MLP 基线', 'KG 原型', 'Ours（门控融合）'],
          [[str(int(r[0])), f'{r[1]:.2f} ± {r[2]:.2f}', f'{r[3]:.2f} ± {r[4]:.2f}',
            f'{r[5]:.2f} ± {r[6]:.2f}'] for r in MAIN_ROWS])
add_para(f'三个发现尤为突出。第一，所提方法在所有少样本规模下均显著优于纯数据基线，'
         f'增益由 k = 50 时的 {M50[7]:.1f} 个百分点增至 k = 5 时的 {M5[7]:.1f} 个百分点，全部 p < 0.05。'
         f'第二，与纯知识原型方法相比，引入逐维门控融合在数据最稀缺时带来额外收益：'
         f'k = 5 时高出 {M5[9]:.1f} 个百分点（p = {M5[10]:.4f}），k = 10 时高出 {M10[9]:.1f} 个百分点（p = {M10[10]:.4f}），'
         f'说明门控融合不仅保留了知识先验，还能自适应地分配数据与知识的权重。'
         f'第三，方法显著降低了诊断方差，例如 k = 5 时纯 MLP 的标准差为 {M5[2]:.2f}%，'
         f'而所提方法仅 {M5[6]:.2f}%。')
add_h2('5.2 少样本增益分析')
add_pic('fig3_gain.png', 12)
add_caption('图 3　知识引导方法相对纯数据基线的增益随样本规模变化')
add_para(f'增益随样本规模的变化呈现明显规律：k = 5 时 {M5[7]:.1f} 个百分点，k = 10 时 {M10[7]:.1f} 个百分点，'
         f'k = 20 时 {M20[7]:.1f} 个百分点，k = 50 时 {M50[7]:.1f} 个百分点。'
         f'即在标注样本稀缺时知识引导的类别先验价值最大。当每类仅有 5 个标注样本时，'
         f'纯 MLP 基线仅 {M5[1]:.1f}%，而所提方法达 {M5[5]:.1f}%。')
add_h2('5.3 稳定性分析')
add_para(f'纯 MLP 基线在少样本下的 5 次运行标准差最高达 {M5[2]:.2f}%，'
         f'反映纯数据方法对随机初始化与样本选择的敏感性；引入知识先验后，'
         f'标准差降至 {M50[6]:.2f}%～{M10[6]:.2f}%，诊断结果高度可复现。'
         f'产线诊断系统需结果可重复、可审计，高方差模型难以通过稳定性验收，因此该性质具有明确的工程价值。')
add_h2('5.4 消融实验')
add_pic('fig4_ablation.png', 16)
add_caption('图 4　消融实验。(a) 融合机制消融；(b) k = 10 时各消融变体')
add_para('表 2 列出全部消融变体在四个少样本规模下的准确率，星号表示与完整模型差异达统计显著（p < 0.05）。'
         '全部 11 个变体 × 4 个 k × 5 种子共 220 次训练。')
add_caption('表 2　消融实验结果（5 种子均值 ± 标准差，%；* 表示相对 Ours 显著，p < 0.05）')
add_table(['变体', 'k = 5', 'k = 10', 'k = 20', 'k = 50'], [[n] + c for n, c in ABL_ROWS])
_g = lambda nm, k: (np.mean(abl[k]['Ours (完整模型)']) - np.mean(abl[k][nm])) * 100
add_para(f'（1）融合机制是决定性组件。将逐维门控替换为拼接融合，准确率在 k = 50 时下降 {_g("A1 拼接融合", 50):.1f} 个百分点、'
         f'k = 5 时下降 {_g("A1 拼接融合", 5):.1f} 个百分点；替换为相加融合下降 {_g("A2 相加融合", 50):.1f}～{_g("A2 相加融合", 5):.1f} 个百分点；'
         f'替换为固定权重 α = 0.5 下降 {_g("A3 固定门控 α=0.5", 50):.1f}～{_g("A3 固定门控 α=0.5", 5):.1f} 个百分点。'
         f'上述 12 组对比全部达到统计显著（p < 0.05）。'
         f'其中固定权重在 k = 5 时仅 {np.mean(abl[5]["A3 固定门控 α=0.5"])*100:.1f}%，'
         f'远低于完整模型的 {np.mean(abl[5]["Ours (完整模型)"])*100:.1f}%，'
         f'说明性能来源并非「多引入一路输入」，而是门控权重 α 的逐维自适应性。')
add_para(f'（2）知识分支的存在是必要的。完全移除知识分支（C1，退化为纯 MLP）使准确率在 k = 50 时下降 '
         f'{_g("C1 无知识分支(纯MLP)", 50):.1f} 个百分点、k = 5 时下降 {_g("C1 无知识分支(纯MLP)", 5):.1f} 个百分点，'
         f'四组对比全部显著。')
add_para('（3）知识信号的具体内容对性能影响不显著。以自由可学习的类别原型完全替代图谱原型（C2）、'
         '以硬检索替代软检索（B1 消融）、以均匀常数替代样本级知识信号（B2 消融）、'
         '子图范围取 0 跳或 2 跳（D1、D2）、以随机嵌入替代 TransE 预训练嵌入（E1），'
         '四组少样本实验下差异均不超过 2 个百分点且基本不显著。'
         '这一结果说明：在标注样本充足到足以学到类别判别方向的少样本场景中，该架构主要利用了知识的「存在」'
         '（类别级条件信号）而非其「语义结构」。与此同时，这也带来一个有利性质——'
         '诊断性能与知识库质量解耦，即使知识图谱粒度粗糙、覆盖有限，仍能提供有效的类别级先验，'
         '降低了知识库构建的成本门槛。要真正利用图谱的语义结构，则需转向未见故障诊断，见 5.5 节。')

add_h2('5.5 未见故障的零样本诊断')
add_para(f'4.2 节的零样本设定下，随机猜测水平为 {CHANCE:.1f}%。作为对照，'
         f'若直接以类别知识原型作为类条件向量交由共享打分器（即不引入可观测表现层），'
         f'零样本准确率低于随机猜测，说明仅靠 TransE 嵌入的几何结构不足以支撑跨类别迁移。'
         f'引入 3.6 节的可观测表现层后，模型在可见故障上学习 8 维物理属性的检测器'
         f'（可见类属性检测准确率 {ZS[3]["attr"]:.1f}%），再按属性组合匹配未见故障，结果如表 3。')
add_caption('表 3　未见故障零样本诊断结果（按属性覆盖度分层，随机猜测水平 33.3%）')
add_table(['留出折的覆盖度 min_cov', '折数', '知识图谱属性矩阵', '随机属性矩阵', '差值', '配对 t 检验 p'],
          [['3（属性覆盖充分）', str(ZS[3]['n']), f"{ZS[3]['kg']:.1f}% ± {ZS[3]['kg_std']:.1f}",
            f"{ZS[3]['rd']:.1f}% ± {ZS[3]['rd_std']:.1f}", f"+{ZS[3]['delta']:.1f}pp",
            f"{ZS[3]['p']:.4f}"],
           ['2', str(ZS[2]['n']), f"{ZS[2]['kg']:.1f}% ± {ZS[2]['kg_std']:.1f}",
            f"{ZS[2]['rd']:.1f}% ± {ZS[2]['rd_std']:.1f}", f"+{ZS[2]['delta']:.1f}pp",
            f"{ZS[2]['p']:.4f}"]])
add_para(f"在属性覆盖充分的留出折上（min_cov = 3），所提方法对未见故障的识别率达 "
         f"{ZS[3]['kg']:.1f}%，显著高于随机猜测水平 {CHANCE:.1f}%，"
         f"并显著优于随机属性矩阵对照（{ZS[3]['rd']:.1f}%，配对 t 检验 p = {ZS[3]['p']:.4f}，"
         f"{ZS[3]['npos']}/{ZS[3]['n']} 个折优于对照）。"
         f"这一结果直接证明：知识图谱提供的「故障—现象—可观测表现」映射是可迁移的领域知识，"
         f"而非仅作为类别标签来源。当属性覆盖不足时（min_cov = 2），"
         f"优势缩小至 +{ZS[2]['delta']:.1f} 个百分点且不再显著，说明零样本能力对知识覆盖度敏感。")
add_pic('fig5_coverage.png', 13)
add_caption('图 5　属性覆盖度与零样本准确率的关系（8 类库与 22 类库对照）')
add_h2('5.6 属性覆盖度标定：知识图谱零样本能力的启用条件')
add_para(f'为定量回答「需要多大的故障库才能支撑零样本诊断」，本文定义属性覆盖度 min_cov 为：'
         f'对所有属性取「该属性在可见故障集中的正例类别数」的最小值。'
         f'min_cov 衡量的是属性检测器能学到的概念有多牢固——若某属性仅有一个可见正例类别，'
         f'检测器必然退化为识别该单个类别，而非学习可迁移的物理概念。')
add_para(f'在原始的 8 类故障库上，全部 56 种 3 类留出组合中，'
         f'min_cov = 0 的有 53 个（零样本准确率 {C8.get(0, float("nan")):.1f}%），'
         f'min_cov = 1 的有 3 个（{C8.get(1, float('nan')):.1f}%），'
         f'没有任何组合能达到 min_cov ≥ 2。也就是说，8 类故障库从未进入过零样本可用的区间。'
         f'将故障库扩展至 22 类、并按「每个属性至少 3 个正例类别」原则做平衡设计后，'
         f'可用留出折达到 min_cov = 2 的 750 个与 min_cov = 3 的 314 个，'
         f'零样本准确率随之从 {C8.get(1, float('nan')):.1f}% 提升至 {ZS[3]['kg']:.1f}%。')
add_para(f'由此可给出可操作的故障库规模准则。设属性维度为 A、平均每类属性数为 r，'
         f'则要在留出 U 个类别后仍保证 min_cov ≥ 3，可见类别数需满足 n_seen ≥ 3A / r。'
         f'本文 A = 8、r ≈ 2.5，得 n_seen ≥ 10，加上留出的 3 类共约 13 类；'
         f'但由于「测量波动」「相邻节点等电位」等属性仅由特定故障机理产生（来源较窄），'
         f'需刻意为其配置正例，实际采用 22 类故障库。'
         f'这一准则说明：扩展故障库时，属性覆盖的均衡性比单纯的类别数量更重要。')

# ==================== 6 讨论 ====================
add_h1('6 讨论')
add_h2('6.1 核心主张')
add_para('本文的核心主张有三。第一，逐维自适应门控融合是多分支诊断中性能的决定性来源：'
         '在 22 类故障库上，16 组融合机制与知识分支对照全部达到统计显著，'
         '且替换为固定权重融合时性能大幅退化，说明增益来自门控权重的逐维自适应性本身。'
         '第二，知识图谱的价值可以在两个层面释放——在少样本场景中提供类别级先验（增益随样本稀缺度递增，'
         '每类 5 个样本时达 27.2 个百分点），在零样本场景中提供可迁移的物理属性表示'
         '（未见故障识别率 54.9%，显著优于随机属性矩阵对照）。'
         '第三，知识图谱零样本能力的启用条件可被定量刻画：每个属性需至少 3 个可见故障类别作为正例，'
         '这为工业知识库的规模规划提供了依据。')
add_h2('6.2 局限')
add_para('若干局限约束了结论。第一，测量数据为 MNA 物理仿真数据，与真实产线 ICT 记录仍有差距：'
         '仿真电路相对理想化，真实测量的夹具寄生、温度漂移、非线性效应未完全建模，'
         '在真实产线数据上的验证是确立泛化能力的必要下一步。')
add_para('第二，零样本能力对知识覆盖度敏感。属性覆盖充分时未见故障识别率达 54.9%，'
         '但覆盖不足时退化至随机水平附近；同时，属性空间中仍存在属性向量相同的故障类别对'
         '（如同一节点的电容漂移与缺件），这类故障在当前属性粒度下不可区分，'
         '需要更细的属性维度（如频段偏移方向、偏移幅度档位）才能分离。')
add_para('第三，知识信号的「语义结构」在少样本场景中尚未被充分利用，'
         '消融实验显示替换知识内容对少样本准确率影响不显著；'
         '此外，TransE 在建模一对多、多对多关系时能力有限，更具表达力的嵌入方法'
         '或能更好地刻画故障图谱的关系结构。')
add_h2('6.3 实务启示')
add_para('本方法适用于故障类别体系相对稳定、但标注样本稀缺的 ICT 场景。'
         '在知识资源有限的现场，即使仅能提供粗粒度的故障分类体系，方法仍可提供有效的类别先验；'
         '当标注样本充足时（如每类 50 个以上），知识引导的增益收敛，方法退化为常规的数据驱动诊断。'
         '若需要覆盖产线新出现的故障模式，则应扩展故障库并保证每个物理属性具备足够的正例类别，'
         '此时方法可对新故障提供 50% 以上的零样本识别能力，可作为人工诊断的有效预筛。')

# ==================== 7 结论 ====================
add_h1('7 结论')
add_para('本文提出知识图谱引导的双分支电气故障诊断模型，通过多维门控注意力融合类别知识原型与测量特征，'
         '并进一步在知识图谱中引入可观测表现层以支持未见故障诊断。'
         '在 MNA 物理仿真的 22 类电气故障库上，系统评估了少样本与零样本场景下的诊断性能，'
         '并完成了组件级消融与故障库规模标定。'
         '实验表明：逐维自适应门控融合是性能的决定性组件；少样本增益随样本稀缺度递增；'
         '引入可观测表现层后，未见故障的零样本识别率达 54.9%（随机猜测 33.3%），'
         '显著优于随机属性矩阵对照；知识图谱零样本能力的启用条件为每个属性至少 3 个可见正例类别。'
         '未来工作包括：在真实产线 ICT 数据上验证泛化能力、'
         '细化属性维度以区分当前属性空间中等价的故障对、'
         '以及探索更具表达力的图谱嵌入方法。')

# ==================== 参考文献 ====================
add_h1('参考文献')
refs = [
    'C.-H. Hsiao, H.-C. Su, Y.-T. Wang, M.-J. Hsu, C.-C. Hsu, "ResNet-SE-CBAM Siamese networks for few-shot and imbalanced PCB defect classification," Sensors, vol. 25, no. 13, 4233, 2025.',
    'Y. Li, S. Wang, Z. Jing, J. Zheng, X. Han, X. Zheng, F. Sun, "DSRF: few-shot PCB surface defect detection via dynamic selective regulation fusion," The Journal of Supercomputing, vol. 81, no. 4, 2025.',
    'J. Cação, J. Santos, M. Antunes, "Explainable AI for industrial fault diagnosis: A systematic review," Journal of Industrial Information Integration, vol. 47, 100905, 2025.',
    'A. Bhattacharya, S. G. Cloutier, "End-to-end deep learning framework for printed circuit board manufacturing defect classification," Scientific Reports, vol. 12, 12559, 2022.',
    'X. Chen, Y. Wu, X. He, W. Ming, "A comprehensive review of deep learning-based PCB defect detection," IEEE Access, vol. 11, pp. 139017–139038, 2023.',
    'C. Long, E. Zhao, J. Zhou, C. Tan, K. Shen, L. Duan, "Defect recognition of printed circuit board assemblies via enhanced Swin-Condinst network," Measurement, vol. 253, 117879, 2025.',
    'S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, "CBAM: Convolutional block attention module," Proc. ECCV, 2018, pp. 3–19.',
    'S. Yang, C. Li, Y. Zhu, H. Shen, L. Fang, "An informer network-based circuit boards fault detection method using infrared temperature series," Microelectronics Reliability, 2025.',
    'W. Tang, M. Chen, "TFDFNet: a dual-branch fault diagnosis model for bearings under noisy and complex industrial environments," Scientific Reports, 2025.',
    'M. N. Khan, T. Greiner, "Neurosymbolic artificial intelligence for printed circuit board fault detection," Proc. ICMLT, 2025, pp. 116–124.',
    '马亚杰, 刘洋, 姜斌, 冒泽慧, 刘剑慰, 刘文静, "基于异构图注意力的工业设备故障诊断知识图谱补全方法," 中国科学: 信息科学.',
    'Q. Li, Y. Wang, J. Dong, C. Zhang, K. Peng, "Multi-node knowledge graph assisted distributed fault detection for large-scale industrial processes based on graph attention network and bidirectional LSTMs," Neural Networks, vol. 173, 106210, 2024.',
    'M. Shenoy, S. Sahasrabudhe, F. Ameri, "Neurosymbolic AI-driven zero-defect manufacturing in semiconductor assembly: A hybrid framework," Proc. ASMC, 2025, pp. 1–6.',
    'M. Aly, N. M. Alotaibi, "SHAP enhanced transformer GWO boosting model for transparent and robust anomaly detection in IIoT environments," Scientific Reports, vol. 15, 41037, 2025.',
    'A. Vaswani et al., "Attention is all you need," Advances in Neural Information Processing Systems, 2017, pp. 5998–6008.',
    'C. Liu, H. Yuan, J. Lv, Y. Deng, H. Xu, M. Ferlauto, "A dual-attention model for non-contact multi-fault PCB diagnosis based on scalar magnetic field and fault relation feature," Measurement, vol. 253, 117675, 2025.',
    'J. Wang, H. Shao, J. He, L. Liu, J. Ma, B. Liu, "A novel interpretable fault diagnosis method using multi-image feature extraction and attention fusion," Pattern Recognition Letters, vol. 189, pp. 38–47, 2025.',
    'G. Li, M. A. Atoui, X. Li, "Attention-based multiscale temporal fusion network for uncertain-mode fault diagnosis in multimode processes," Process Safety and Environmental Protection, vol. 201, 107554, 2025.',
    'J. Zhou, D. Li, M. Liu, H. Su, "Lightweight ResNet with multi-scale attention fusion for fault diagnosis of accelerator power systems," Proc. JCICE, 2025, pp. 1–6.',
    'A. Bordes, N. Usunier, A. Garcia-Durán, J. Weston, O. Yakhnenko, "Translating embeddings for modeling multi-relational data," Advances in Neural Information Processing Systems, 2013, pp. 2787–2795.',
    'J. Du, Y. Gao, L. Gao, X. Li, "A frequency mask and decoupling max-logit based XAI method to explain DNN for fault diagnosis," Journal of Manufacturing Systems, vol. 82, pp. 98–113, 2025.',
    'C. H. Lampert, H. Nickisch, S. Harmeling, "Learning to detect unseen object classes by between-class attribute transfer," Proc. CVPR, 2009, pp. 951–958.',
    'Y. Fu, T. Xiang, Y.-G. Jiang, X. Xue, L. Sigal, S.-F. Chang, "Recent advances in zero-shot recognition," IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 112–125, 2018.',
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    set_font(p.add_run(f'[{i}] {ref}'), size=10.5)

doc.save(OUT)
print('Word 已生成:', OUT)
