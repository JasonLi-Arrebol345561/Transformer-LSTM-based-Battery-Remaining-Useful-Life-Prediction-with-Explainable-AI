# 基于 Transformer/LSTM 的锂电池寿命预测与可解释性分析系统

基于 MIT-Stanford 124 颗 A123 LFP 18650 电池快充老化数据集，构建**统计分析 → 机器学习预测 → 可解释性验证**的完整研究闭环。

## 项目结构

```
BatteryLife-AI/
├── README.md                          # 本文件
├── Battery Analysis Report.md         # 详细分析报告
│
├── data/                              # 数据文件
│   ├── MIT-Stanford Battery Dataset_cleaned.mat  # 原始数据 (7.4 GB, HDF5 v7.3)
│   ├── battery_analysis_results.mat              # 预处理后的分析结果
│   └── B.docx                                   # 补充文档
│
├── code/                              # 代码文件（按执行顺序排列）
│   ├── battery_analysis.py            # ① 数据预处理 + 箱线图
│   ├── spearman_analysis.py           # ② Spearman 秩相关 + SOC 区间分析
│   ├── feature_importance.py          # ③ 交互效应 + 综合排序
│   ├── model_comparison.py            # ④ 四模型对比 (Ridge/RF/LSTM/Transformer)
│   ├── feature_ablation.py            # ⑤ Transformer 特征消融实验
│   ├── holdout_shap_analysis.py       # ⑥ 留出测试集 + SHAP 可解释
│   └── final_figures.py               # ⑦ 论文终图 (预测散点图 + 注意力图)
│
├── figures/                           # 输出图表 (PNG, 150 dpi)
│   ├── boxplots.png                   # 循环寿命与充电时间箱线图
│   ├── spearman_bar.png               # 各 SOC 区间 Spearman ρ 柱状图
│   ├── spearman_scatter.png           # C1/Q1/C2 vs 循环寿命散点图
│   ├── feature_importance.png         # 因素交互作用排序
│   ├── feature_ablation.png           # Feature Ablation R² 柱状图
│   ├── pred_vs_true.png              # 预测 vs 真实循环寿命散点图
│   ├── shap_summary.png              # SHAP 特征重要性
│   ├── shap_waterfall.png            # SHAP 瀑布图（单电池）
│   ├── attention_map.png             # Transformer 注意力热力图
│   └── attention_profile.png         # 注意力时序分布
│
└── Paper/                             # LaTeX 论文
    ├── battery_life_prediction.tex    # 论文源文件
    └── battery_life_prediction.pdf    # 编译后的 PDF
```

## 环境依赖

| 依赖 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.11 (CUDA 12.8) |
| scikit-learn | 1.9 |
| NumPy | ≥1.24 |
| SciPy | ≥1.11 |
| h5py | ≥3.9 |
| SHAP | ≥0.44 |
| Matplotlib | ≥3.7 |

## 快速开始

按以下顺序执行脚本：

```bash
cd code

# ① 数据预处理（生成 battery_analysis_results.mat 和箱线图）
python battery_analysis.py

# ② Spearman 相关性分析
python spearman_analysis.py

# ③ 交互效应与综合排序
python feature_importance.py

# ④ 四模型对比
python model_comparison.py

# ⑤ 特征消融实验
python feature_ablation.py

# ⑥ 留出测试集 + SHAP 分析
python holdout_shap_analysis.py

# ⑦ 论文终图
python final_figures.py
```

> **注意**：脚本 ④–⑦ 需要 CUDA GPU。如无 GPU，模型训练会显著变慢，但 CPU 模式仍可运行。

## 编译论文

```bash
cd Paper
xelatex battery_life_prediction.tex
xelatex battery_life_prediction.tex   # 第二次以解析交叉引用
```

## 核心发现

1. **Q₁（阶段切换 SOC）是最关键的策略参数** — Spearman ρ = +0.317，且是三个最强交互效应的枢纽变量
2. **SOC 40–80% 是高倍率充电的"危险区间"** — 该区间 C-rate 与寿命的 ρ = −0.340
3. **交互效应主导了充电策略的综合影响** — 综合排序前三名均为交互效应，C₁ × Q₁ 交互强度达 0.825
4. **Transformer 预测 R² = 0.878**（留出测试集），显著优于 Ridge (−0.09)、RF (0.49)、LSTM (0.50)
5. **SHAP 和 Attention 分析验证了模型决策的物理合理性** — 内阻特征和 Q₁ 同时是统计显著和模型重要的变量

## 数据来源

Severson, K. A., et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy, 2019, 4: 383–391.

## 许可证

本项目仅用于学术研究目的。
