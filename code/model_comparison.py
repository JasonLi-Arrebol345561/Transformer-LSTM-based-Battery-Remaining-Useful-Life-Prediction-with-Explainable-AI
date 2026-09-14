# -*- coding: utf-8 -*-
"""
=============================================================================
第一阶段：公平对比 — 四模型 + 朴素基线，多 seed 报告均值±标准差
=============================================================================
  模型 — Ridge / Random Forest / LSTM / Transformer，加 Dummy(均值) 朴素基线。
  Ridge/RF 输入展平的原始序列 (100×9=900 维)；LSTM/Transformer 输入 (100×9) 时序，
  保证四者信息量对等（公平对比）。

  复现性：DataLoader 使用每折独立的 fold_generator(seed, fold)，不再复用全局共享
  generator；LSTM/Transformer 跑 N_SEEDS 个 seed，报告 (seed×fold) 的均值±标准差。
  指标：MAE / RMSE / R² / MAPE。

  预测目标：cycle_life（真实 EOL 标签）。
=============================================================================
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import common as C
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset

C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()

N_SEEDS = 5
seeds = [C.SEED + i for i in range(N_SEEDS)]

# ============================================================
# 1. 读取数据（内部做电池顺序校验）
# ============================================================
FEATURES, y_life, meta = C.load_all_features(with_dqdv=False)
X_seq = FEATURES['B']                        # (124, 100, 9) Feature B
X_flat = C.flatten_sequence(X_seq)           # (124, 900) 展平

N_BAT, SEQ_LEN, N_FEAT = X_seq.shape
print(f'数据: {N_BAT} 电池, 序列 {X_seq.shape}, 展平 {X_flat.shape}')
print(f'cycle_life 范围: {y_life.min():.0f} - {y_life.max():.0f}, 均值: {y_life.mean():.1f}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device}, 种子数: {N_SEEDS}')

kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
splits = list(kf.split(np.arange(N_BAT)))   # 固定划分，所有 seed 共享


def eval_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE': C.mape(y_true, y_pred),
    }


def summarize(vals):
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}


results = {}


def record(name, m_list):
    results[name] = {k: summarize([m[k] for m in m_list]) for k in ['MAE', 'RMSE', 'R2', 'MAPE']}


# ============================================================
# 2. 朴素基线：预测训练集均值
# ============================================================
print('\n' + '=' * 60)
print('基线: Dummy (预测均值)')
print('=' * 60)
dummy = []
for fold, (tr, va) in enumerate(splits):
    y_pred = np.full(len(va), y_life[tr].mean())
    m = eval_metrics(y_life[va], y_pred)
    dummy.append(m)
    print(f'  Fold {fold + 1}: MAE={m["MAE"]:.0f}, R²={m["R2"]:.3f}, MAPE={m["MAPE"]:.1f}%')
record('Dummy(mean)', dummy)
print(f'  平均: R²={results["Dummy(mean)"]["R2"]["mean"]:.3f}')

# ============================================================
# 3. Ridge（展平原始序列，确定性）
# ============================================================
print('\n' + '=' * 60)
print('模型 A: Ridge 回归 (展平原始序列, 900 维)')
print('=' * 60)
m_a = []
for fold, (tr, va) in enumerate(splits):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_flat[tr])
    X_val_s = scaler.transform(X_flat[va])
    model = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=3, scoring='r2')
    model.fit(X_tr_s, y_life[tr])
    m = eval_metrics(y_life[va], model.predict(X_val_s))
    m_a.append(m)
    print(f'  Fold {fold + 1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R²={m["R2"]:.3f}, MAPE={m["MAPE"]:.1f}%')
record('Ridge', m_a)

# ============================================================
# 4. Random Forest（展平原始序列，确定性）
# ============================================================
print('\n' + '=' * 60)
print('模型 B: Random Forest (展平原始序列, 900 维)')
print('=' * 60)
m_b = []
for fold, (tr, va) in enumerate(splits):
    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                  random_state=C.SEED, n_jobs=-1)
    model.fit(X_flat[tr], y_life[tr])
    m = eval_metrics(y_life[va], model.predict(X_flat[va]))
    m_b.append(m)
    print(f'  Fold {fold + 1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R²={m["R2"]:.3f}, MAPE={m["MAPE"]:.1f}%')
record('RF', m_b)


# ============================================================
# 5. 序列模型（LSTM / Transformer，多 seed）
# ============================================================
def run_seq(ModelClass, lr, name, tag):
    m_list = []
    for seed in seeds:
        for fold, (tr, va) in enumerate(splits):
            X_tr, X_val = X_seq[tr], X_seq[va]
            y_tr, y_val = y_life[tr], y_life[va]
            # 每折独立、确定性的随机源（模型初始化 + DataLoader 洗牌）
            torch.manual_seed(seed * 1000 + fold)
            g = C.fold_generator(seed, fold, tag)

            X_tr_s, X_val_s = C.standardize_sequences(X_tr, X_val)
            y_mean, y_std = y_tr.mean(), y_tr.std()
            tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor((y_tr - y_mean) / y_std))
            va_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
            tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, generator=g)
            va_dl = DataLoader(va_ds, batch_size=64)

            model = ModelClass(input_dim=X_tr.shape[2])
            model = C.train_model(model, tr_dl, va_dl, epochs=200, lr=lr, device=device)
            model.eval()
            with torch.no_grad():
                y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
            y_pred = y_pred_s * y_std + y_mean
            m_list.append(eval_metrics(y_val, y_pred))
        print(f'  {name} seed {seed}: 平均 R²={np.mean([m["R2"] for m in m_list[-5:]]):.3f}')
    record(name, m_list)
    r = results[name]
    print(f'  {name} 汇总: R²={r["R2"]["mean"]:.3f}±{r["R2"]["std"]:.3f}, MAPE={r["MAPE"]["mean"]:.1f}±{r["MAPE"]["std"]:.1f}%')


print('\n' + '=' * 60)
print('模型 C: LSTM (原始时序 100×9)')
print('=' * 60)
run_seq(LSTMPredictor, 1e-3, 'LSTM', tag=1)

print('\n' + '=' * 60)
print('模型 D: Transformer (原始时序 100×9)')
print('=' * 60)
run_seq(TransformerPredictor, 5e-4, 'Transformer', tag=2)

# ============================================================
# 6. 汇总 + 落盘
# ============================================================
print('\n' + '=' * 60)
print(f'模型对比汇总（{"5 折" if False else "5 折"}均值 ± 标准差，序列模型跨 {N_SEEDS} seed）')
print('=' * 60)
print(f'{"模型":<14} {"MAE":>14} {"RMSE":>14} {"R²":>16} {"MAPE(%)":>14}')
print('-' * 72)
for name in ['Dummy(mean)', 'Ridge', 'RF', 'LSTM', 'Transformer']:
    r = results[name]
    print(f'{name:<14} {r["MAE"]["mean"]:>7.0f}±{r["MAE"]["std"]:.0f}  '
          f'{r["RMSE"]["mean"]:>7.0f}±{r["RMSE"]["std"]:.0f}  '
          f'{r["R2"]["mean"]:>8.3f}±{r["R2"]["std"]:.3f}  '
          f'{r["MAPE"]["mean"]:>8.1f}±{r["MAPE"]["std"]:.1f}')

C.save_metrics('model_comparison', {
    'input': 'Feature B (100×9), 展平 900 维给 Ridge/RF; 序列模型跨 5 seed',
    'n_seeds': N_SEEDS,
    'results': results,
})
print('\n全部完成!')
