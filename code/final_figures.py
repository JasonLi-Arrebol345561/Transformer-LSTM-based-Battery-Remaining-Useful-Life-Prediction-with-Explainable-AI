# -*- coding: utf-8 -*-
"""
=============================================================================
论文收尾: Prediction scatter + 汇总表（从 metrics.json 读取，不再硬编码）
=============================================================================
  1. 预测 vs 真实散点图（5 折交叉验证汇总, Feature D, figures/pred_vs_true.png）
  2. 控制台输出汇总表：表2 模型对比 / 表3 特征消融 / 表4 留出测试集
     —— 数值全部从 results/metrics.json 读取，保证图/表/报告一致。

  说明：注意力图已移至 explain_transformer.py（rollout + IG + 置换，带基线）。
=============================================================================
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import common as C
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(C.fig_dir(), exist_ok=True)

g = C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# 1. 5 折 CV 收集预测 → 散点图
# ============================================================
FEATURES, y_life, meta = C.load_all_features()
X_seq = FEATURES['D']

kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
all_y_true, all_y_pred = [], []

for fold, (tr, va) in enumerate(kf.split(X_seq)):
    X_tr, X_val = X_seq[tr], X_seq[va]
    y_tr, y_val = y_life[tr], y_life[va]
    X_tr_s, X_val_s = C.standardize_sequences(X_tr, X_val)
    y_mean, y_std = y_tr.mean(), y_tr.std()
    tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor((y_tr - y_mean) / y_std))
    va_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, generator=g)
    va_dl = DataLoader(va_ds, batch_size=64)
    model = TransformerPredictor(input_dim=X_tr.shape[2])
    model = C.train_model(model, tr_dl, va_dl, epochs=200, lr=5e-4, device=device)
    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred_s * y_std + y_mean)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
r2_all = r2_score(all_y_true, all_y_pred)
mae_all = mean_absolute_error(all_y_true, all_y_pred)
rmse_all = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
print(f'5 折 CV 汇总: R²={r2_all:.3f}, MAE={mae_all:.0f}, RMSE={rmse_all:.0f}')

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(all_y_true, all_y_pred, c='#2E86AB', edgecolors='#1A5276', alpha=0.7, s=60, zorder=3)
lims = [min(all_y_true.min(), all_y_pred.min()), max(all_y_true.max(), all_y_pred.max())]
ax.plot(lims, lims, '--', color='#FF6B6B', linewidth=2, label='理想预测', zorder=2)
ax.fill_between(lims, [l - rmse_all for l in lims], [l + rmse_all for l in lims],
                alpha=0.1, color='#FF6B6B', label=f'±1 RMSE ({rmse_all:.0f} cycles)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('真实循环寿命', fontsize=13)
ax.set_ylabel('预测循环寿命', fontsize=13)
ax.set_title(f'预测 vs 真实循环寿命（早期预测）\n(5 折交叉验证, R²={r2_all:.3f}, MAE={mae_all:.0f})',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'pred_vs_true.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 2. 汇总表（从 metrics.json 读取）
# ============================================================
print('\n' + '=' * 60)
print('汇总表（数值来自 results/metrics.json）')
print('=' * 60)

mc = C.load_metrics('model_comparison')
if mc:
    print('\n表2: 模型对比（5 折交叉验证均值 ± 标准差）')
    print(f'{"模型":<15} {"MAE":>16} {"RMSE":>16} {"R²":>18}')
    print('-' * 65)
    for name in ['Ridge', 'RF', 'LSTM', 'Transformer']:
        r = mc['results'][name]
        print(f'{name:<15} {r["MAE"]["mean"]:>7.0f}±{r["MAE"]["std"]:.0f}  '
              f'{r["RMSE"]["mean"]:>7.0f}±{r["RMSE"]["std"]:.0f}  '
              f'{r["R2"]["mean"]:>8.3f}±{r["R2"]["std"]:.3f}')

fa = C.load_metrics('feature_ablation')
if fa:
    print('\n表3: 特征消融实验（Transformer, 5 折交叉验证）')
    print(f'{"阶段":<16} {"维度":<5} {"R²":<18} {"ΔR²":<8}')
    print('-' * 50)
    stages = fa['stages']
    deltas = fa['deltas']
    for k in ['A (基础退化)', 'B (+策略)', 'C (+SOC暴露)', 'D (+dQ/dV)']:
        r = stages[k]['R2']
        print(f'{k:<16} {stages[k]["dim"]:<5} {r["mean"]:.3f}±{r["std"]:.3f}   +{deltas[k]:.3f}')

ho = C.load_metrics('holdout')
if ho:
    print('\n表4: 留出测试集性能（多次分层划分）')
    print('-' * 40)
    print(f'  协议    : {ho["protocol"]} = {ho["n_repeats"]}')
    print(f'  测试电池: 平均 {ho["n_test_mean"]:.0f} 颗')
    print(f'  R²       = {ho["R2"]["mean"]:.3f} ± {ho["R2"]["std"]:.3f}')
    print(f'  MAE      = {ho["MAE"]["mean"]:.0f} ± {ho["MAE"]["std"]:.0f} cycles')
    print(f'  RMSE     = {ho["RMSE"]["mean"]:.0f} ± {ho["RMSE"]["std"]:.0f} cycles')

print('\n全部完成!')
