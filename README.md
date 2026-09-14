# 基于 Transformer 的锂电池早期循环寿命预测与可解释性分析

基于 MIT-Stanford 124 颗 A123 LFP 18650 电池快充老化数据集，构建**统计分析 → 机器学习预测 → 可解释性验证**的完整研究闭环。任务定义为**早期循环寿命预测**（early cycle-life prediction）：用前 100 个循环的测量数据预测电池总循环寿命。

## 项目结构

```
BatteryLife-AI/
├── README.md                          # 本文件
├── Battery Analysis Report.md         # 详细分析报告
│
├── data/                              # 数据文件
│   ├── MIT-Stanford Battery Dataset_cleaned.mat  # 原始数据 (7.4 GB, HDF5 v7.3)
│   └── battery_analysis_results.mat              # 预处理后的分析结果
│
├── code/                              # 代码文件（按执行顺序排列）
│   ├── common.py                      # 共享模块（路径/加载/顺序校验/特征/模型/种子）
│   ├── battery_analysis.py            # ① 数据预处理 + 箱线图
│   ├── spearman_analysis.py           # ② Spearman 秩相关 + SOC 区间分析（FDR 校正）
│   ├── feature_importance.py          # ③ 主效应 + 交互效应（Fisher z 检验）
│   ├── model_comparison.py            # ④ 四模型公平对比 (Ridge/RF/LSTM/Transformer)
│   ├── feature_ablation.py            # ⑤ Transformer 特征消融实验
│   ├── holdout_eval.py                # ⑥ 留出测试集评估（多次分层划分）
│   ├── explain_transformer.py         # ⑦ Transformer 可解释性（IG/置换/rollout）
│   └── final_figures.py               # ⑧ 论文终图 (预测散点图 + 汇总表)
│
├── results/                           # metrics.json（图/表/报告数字的单一来源）
│   └── metrics.json
│
├── figures/                           # 输出图表 (PNG, 150 dpi)
│   ├── boxplots.png                   # 循环寿命与充电时间箱线图
│   ├── spearman_bar.png               # 各 SOC 区间 Spearman ρ 柱状图
│   ├── spearman_scatter.png           # C1/Q1/C2 vs 循环寿命散点图
│   ├── feature_importance.png         # 主效应与交互效应（含显著性）
│   ├── feature_ablation.png           # Feature Ablation R² 柱状图
│   ├── pred_vs_true.png               # 预测 vs 真实循环寿命散点图
│   ├── ig_importance.png              # Transformer Integrated Gradients 特征重要性
│   ├── permutation_importance.png     # Transformer 置换重要性
│   ├── attention_rollout.png          # 注意力 rollout（训练模型 vs 随机基线）
│   └── attention_profile.png          # rollout 时间步贡献分布
│
└── Paper/                             # LaTeX 论文
    ├── battery_life_prediction.tex    # 论文源文件
    └── battery_life_prediction.pdf    # 编译后的 PDF
```

## 环境依赖

| 依赖 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | ≥2.0（CUDA 版实测 2.11） |
| scikit-learn | ≥1.3 |
| NumPy | ≥1.24 |
| SciPy | ≥1.11 |
| h5py | ≥3.9 |
| Matplotlib | ≥3.7 |

完整依赖见 [`requirements.txt`](requirements.txt)。

## 快速开始

按以下顺序执行脚本：

```bash
cd code

# ① 数据预处理（生成 battery_analysis_results.mat 和箱线图）
python battery_analysis.py

# ② Spearman 相关性分析
python spearman_analysis.py

# ③ 主效应与交互效应（Fisher z 检验 + FDR）
python feature_importance.py

# ④ 四模型公平对比
python model_comparison.py

# ⑤ 特征消融实验
python feature_ablation.py

# ⑥ 留出测试集评估（多次分层划分）
python holdout_eval.py

# ⑦ Transformer 可解释性（IG / 置换 / rollout，带基线）
python explain_transformer.py

# ⑧ 论文终图
python final_figures.py
```

> **注意**：本项目为 124 个样本的小规模序列，**CPU 即可完成全部训练**，无需 GPU。脚本 ① 需读取 7.36 GB 原始 .mat，耗时主要在此 I/O。

## 编译论文

```bash
cd Paper
xelatex battery_life_prediction.tex
xelatex battery_life_prediction.tex   # 第二次以解析交叉引用
```

## 核心发现

1. **Q₁（阶段切换 SOC）是最关键的可解释策略参数** — Spearman ρ = +0.318（FDR 显著）；但「单次平均充电时间」是更强的主效应（ρ = −0.528）。
2. **SOC 40–80% 是高倍率充电的"危险区间"** — 40–60% 与 60–80% 区间的 C-rate 与寿命 ρ 分别为 −0.287、−0.330（FDR 显著）；低 SOC 区间不显著。
3. **交互效应经 Fisher r-to-z 检验**：C₁ × Q₁（|Δρ|=0.824，p≈5e-7）与 Q₁ × C₂（|Δρ|=0.468，p≈0.005）显著，其余交互不显著。
4. **公平对比**（Ridge/RF 与 LSTM/Transformer 输入同一套原始序列，序列模型跨 5 seed）：Ridge R² = 0.577±0.097（不再为负），Transformer 最优 R² = 0.834±0.067；留出测试集（10 次分层划分）R² = 0.773±0.175、MAPE 18.8%（约为 Severson 2019 同数据集 9.1% 的 2 倍，故不宣称「高精度」）。
5. **特征消融经配对 Wilcoxon 检验**：仅策略参数（B）增益显著（p≈1.5e-6），SOC 区间倍率（C）无增益（p=0.12），dQ/dV（D）边缘（p=0.08）。
6. **可解释性落在 Transformer 本身**（Integrated Gradients / 置换重要性 / 注意力 rollout），并附随机初始化与时间轴打乱基线；时间打乱几乎不降 R²，说明模型依赖退化水平而非时序顺序。

## 数据来源

Severson, K. A., et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy, 2019, 4: 383–391.

## 许可证

本项目采用 [MIT License](LICENSE)。数据来自 Severson et al. (2019)，使用时请遵循原始数据集的使用条款。
