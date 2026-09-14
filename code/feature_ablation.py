# -*- coding: utf-8 -*-
"""
=============================================================================
第二阶段：Feature Ablation — Transformer 逐层叠加特征（多 seed + Wilcoxon）
=============================================================================
  量化每一类物理特征对早期循环寿命预测的贡献。
  特征递进（统一由 common.load_all_features 构建）：
    A (基础退化) : 6 维动态 SOH/IR/Tavg/QCharge/QDischarge/charge_time
    B (+策略)     : 9 维 = A + 静态 C1/Q1/C2
    C (+SOC暴露)  : 13 维 = B + 4 个 SOC 区间等效 C-rate（宽度加权）
    D (+dQ/dV)    : 18 维 = C + 5 个 dQ/dV 曲线统计量

  复现性：每折用独立 fold_generator(seed, fold)；跑 N_SEEDS 个 seed。
  逐层增量用**配对 Wilcoxon 检验**判断是否显著（只对 p<0.05 才写「有增益」）。
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
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader, TensorDataset

C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()

N_SEEDS = 5
seeds = [C.SEED + i for i in range(N_SEEDS)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device}, 种子数: {N_SEEDS}')

FEATURES, y_life, meta = C.load_all_features()
N_BAT = C.N_BAT

kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
splits = list(kf.split(np.arange(N_BAT)))

STAGE_LABELS = {'A': 'A (基础退化)', 'B': 'B (+策略)', 'C': 'C (+SOC暴露)', 'D': 'D (+dQ/dV)'}
STAGE_NAMES = ['A', 'B', 'C', 'D']

# 收集每个阶段的 (seed×fold) 结果；各阶段按同一 (seed, fold) 顺序，便于配对检验
r2_all = {k: [] for k in STAGE_NAMES}
mae_all = {k: [] for k in STAGE_NAMES}
rmse_all = {k: [] for k in STAGE_NAMES}

for key in STAGE_NAMES:
    X = FEATURES[key]
    in_dim = X.shape[2]
    print(f'\n{"=" * 60}')
    print(f'Feature {STAGE_LABELS[key]}  (dim={in_dim})')
    print(f'{"=" * 60}')

    for seed in seeds:
        for fold, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_val = X[tr_idx], X[va_idx]
            y_tr, y_val = y_life[tr_idx], y_life[va_idx]
            torch.manual_seed(seed * 1000 + fold)          # 确定性模型初始化
            g = C.fold_generator(seed, fold)               # 独立 DataLoader 生成器

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

            r2_all[key].append(float(r2_score(y_val, y_pred)))
            mae_all[key].append(float(mean_absolute_error(y_val, y_pred)))
            rmse_all[key].append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

    print(f'  {STAGE_LABELS[key]}: R²={np.mean(r2_all[key]):.3f} ± {np.std(r2_all[key]):.3f}')

# ============================================================
# 汇总 + 逐层 Wilcoxon 检验
# ============================================================
all_results = {}
for key in STAGE_NAMES:
    all_results[key] = {
        'dim': FEATURES[key].shape[2],
        'MAE': {'mean': float(np.mean(mae_all[key])), 'std': float(np.std(mae_all[key]))},
        'RMSE': {'mean': float(np.mean(rmse_all[key])), 'std': float(np.std(rmse_all[key]))},
        'R2': {'mean': float(np.mean(r2_all[key])), 'std': float(np.std(r2_all[key]))},
    }

# 逐层增量（A 无增量；B/C/D 用配对 Wilcoxon 检验 + 中位数 ΔR²）
deltas = {STAGE_LABELS['A']: None}
for i in range(1, len(STAGE_NAMES)):
    cur, prev = STAGE_NAMES[i], STAGE_NAMES[i - 1]
    diff = np.array(r2_all[cur]) - np.array(r2_all[prev])
    try:
        _, p = wilcoxon(r2_all[cur], r2_all[prev])
    except ValueError:
        p = float('nan')
    deltas[STAGE_LABELS[cur]] = {'delta_median': float(np.median(diff)), 'wilcoxon_p': float(p)}

print('\n' + '=' * 60)
print('消融实验结果汇总（逐层 Wilcoxon 检验）')
print('=' * 60)
print(f'{"阶段":<18} {"维度":<5} {"R²":<18} {"ΔR²(中位)":<12} {"Wilcoxon p":<12} {"结论"}')
print('-' * 72)
for i, k in enumerate(STAGE_NAMES):
    lbl = STAGE_LABELS[k]
    r2 = all_results[k]['R2']
    if i == 0:
        print(f'{lbl:<18} {all_results[k]["dim"]:<5} {r2["mean"]:.3f}±{r2["std"]:.3f}  {"—":<12} {"—":<12} 基线')
    else:
        d = deltas[lbl]
        sig = '显著' if d['wilcoxon_p'] < 0.05 else '不显著'
        print(f'{lbl:<18} {all_results[k]["dim"]:<5} {r2["mean"]:.3f}±{r2["std"]:.3f}  '
              f'{d["delta_median"]:>+.3f}     {d["wilcoxon_p"]:<12.3g} {sig}')

# ============================================================
# 图表
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(C.fig_dir(), exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

r2_values = [all_results[k]['R2']['mean'] for k in STAGE_NAMES]
r2_stds = [all_results[k]['R2']['std'] for k in STAGE_NAMES]

fig, ax = plt.subplots(figsize=(9, 5.5))
short_names = ['A\n基础退化', 'B\n+策略', 'C\n+SOC暴露', 'D\n+dQ/dV']
bars = ax.bar(short_names, r2_values, color=['#95A5A6', '#2E86AB', '#4ECDC4', '#FF6B6B'],
              edgecolor='white', linewidth=1.5, width=0.55)
for bar, r2, std in zip(bars, r2_values, r2_stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'R²={r2:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for i in range(1, len(STAGE_NAMES)):
    lbl = STAGE_LABELS[STAGE_NAMES[i]]
    d = deltas[lbl]
    mid_x = (bars[i - 1].get_x() + bars[i - 1].get_width() / 2 +
             bars[i].get_x() + bars[i].get_width() / 2) / 2
    star = '*' if d['wilcoxon_p'] < 0.05 else ' (n.s.)'
    ax.annotate(f'{d["delta_median"]:+.3f}{star}', xy=(mid_x, max(r2_values[i - 1], r2_values[i]) + 0.06),
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
ax.set_ylabel('R² 分数', fontsize=13)
ax.set_title(f'特征消融实验: Transformer 逐层特征叠加\n(5 折 × {N_SEEDS} seed, * = Wilcoxon p<0.05)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, max(r2_values) + 0.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'feature_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()

C.save_metrics('feature_ablation', {
    'n_seeds': N_SEEDS,
    'stages': {STAGE_LABELS[k]: all_results[k] for k in STAGE_NAMES},
    'deltas': deltas,
})
print('\n全部完成!')
