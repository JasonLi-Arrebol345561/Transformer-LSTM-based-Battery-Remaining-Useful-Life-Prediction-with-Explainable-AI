# -*- coding: utf-8 -*-
"""
=============================================================================
第二阶段：Feature Ablation — Transformer 逐层叠加特征
=============================================================================
  量化每一类物理特征对早期循环寿命预测的贡献（R² 提升）。
  特征递进（统一由 common.load_all_features 构建）：
    A (基础退化) : 6 维动态 SOH/IR/Tavg/QCharge/QDischarge/charge_time
    B (+策略)     : 9 维 = A + 静态 C1/Q1/C2
    C (+SOC暴露)  : 13 维 = B + 4 个 SOC 区间等效 C-rate（宽度加权）
    D (+dQ/dV)    : 18 维 = C + 5 个 dQ/dV 曲线统计量

  5 折交叉验证（固定 seed），报告 R² 均值±标准差，写入 results/metrics.json。
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

g = C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device}')

FEATURES, y_life, meta = C.load_all_features()
N_BAT = C.N_BAT

kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
splits = list(kf.split(np.arange(N_BAT)))

STAGE_LABELS = {'A': 'A (基础退化)', 'B': 'B (+策略)', 'C': 'C (+SOC暴露)', 'D': 'D (+dQ/dV)'}
STAGE_NAMES = ['A', 'B', 'C', 'D']

all_results = {}

for key in STAGE_NAMES:
    X = FEATURES[key]
    in_dim = X.shape[2]
    print(f'\n{"=" * 60}')
    print(f'Feature {STAGE_LABELS[key]}  (dim={in_dim})')
    print(f'{"=" * 60}')

    metrics = {'MAE': [], 'RMSE': [], 'R2': []}
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_val = X[tr_idx], X[va_idx]
        y_tr, y_val = y_life[tr_idx], y_life[va_idx]

        X_tr_s, X_val_s = C.standardize_sequences(X_tr, X_val)
        y_mean, y_std = y_tr.mean(), y_tr.std()
        tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor((y_tr - y_mean) / y_std))
        va_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
        tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, generator=g)
        va_dl = DataLoader(va_ds, batch_size=64)

        model = TransformerPredictor(input_dim=in_dim)
        model = C.train_model(model, tr_dl, va_dl, epochs=200, lr=5e-4, device=device)
        model.eval()
        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
        y_pred = y_pred_s * y_std + y_mean

        metrics['MAE'].append(float(mean_absolute_error(y_val, y_pred)))
        metrics['RMSE'].append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        metrics['R2'].append(float(r2_score(y_val, y_pred)))
        print(f'  Fold {fold + 1}: MAE={metrics["MAE"][-1]:.0f}, '
              f'RMSE={metrics["RMSE"][-1]:.0f}, R²={metrics["R2"][-1]:.3f}')

    all_results[key] = {
        'dim': in_dim,
        'MAE': {'mean': float(np.mean(metrics['MAE'])), 'std': float(np.std(metrics['MAE']))},
        'RMSE': {'mean': float(np.mean(metrics['RMSE'])), 'std': float(np.std(metrics['RMSE']))},
        'R2': {'mean': float(np.mean(metrics['R2'])), 'std': float(np.std(metrics['R2']))},
    }
    print(f'  平均 R² = {all_results[key]["R2"]["mean"]:.3f} ± {all_results[key]["R2"]["std"]:.3f}')

# ============================================================
# 汇总 + 图表
# ============================================================
r2_values = [all_results[k]['R2']['mean'] for k in STAGE_NAMES]
r2_stds = [all_results[k]['R2']['std'] for k in STAGE_NAMES]
dims = [all_results[k]['dim'] for k in STAGE_NAMES]
r2_deltas = [r2_values[0]] + [r2_values[i] - r2_values[i - 1] for i in range(1, len(STAGE_NAMES))]

print('\n' + '=' * 60)
print('消融实验结果汇总')
print('=' * 60)
print(f'{"阶段":<18} {"维度":<5} {"R²":<16} {"ΔR²":<8}')
print('-' * 50)
for i, k in enumerate(STAGE_NAMES):
    print(f'{STAGE_LABELS[k]:<18} {dims[i]:<5} {r2_values[i]:.3f} ± {r2_stds[i]:.3f}  +{r2_deltas[i]:.3f}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(C.fig_dir(), exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(9, 5.5))
short_names = ['A\n基础退化', 'B\n+策略', 'C\n+SOC暴露', 'D\n+dQ/dV']
bars = ax.bar(short_names, r2_values, color=['#95A5A6', '#2E86AB', '#4ECDC4', '#FF6B6B'],
              edgecolor='white', linewidth=1.5, width=0.55)
for bar, r2, std in zip(bars, r2_values, r2_stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'R²={r2:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for i in range(1, len(STAGE_NAMES)):
    mid_x = (bars[i - 1].get_x() + bars[i - 1].get_width() / 2 +
             bars[i].get_x() + bars[i].get_width() / 2) / 2
    ax.annotate(f'+{r2_deltas[i]:.3f}', xy=(mid_x, max(r2_values[i - 1], r2_values[i]) + 0.06),
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
ax.set_ylabel('R² 分数', fontsize=13)
ax.set_title('特征消融实验: Transformer 逐层特征叠加\n(5 折交叉验证均值 ± 标准差)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(r2_values) + 0.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'feature_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()

C.save_metrics('feature_ablation', {
    'stages': {STAGE_LABELS[k]: all_results[k] for k in STAGE_NAMES},
    'deltas': {STAGE_LABELS[k]: float(r2_deltas[i]) for i, k in enumerate(STAGE_NAMES)},
})
print('\n全部完成!')
