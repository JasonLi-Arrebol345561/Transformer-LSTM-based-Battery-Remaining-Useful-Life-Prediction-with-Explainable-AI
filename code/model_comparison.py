# -*- coding: utf-8 -*-
"""
=============================================================================
第一阶段：公平对比 — 四个模型在同一原始序列信息量下
=============================================================================
  模型 A — Ridge 线性回归     (展平的原始序列 100×9=900 维)
  模型 B — Random Forest      (展平的原始序列 900 维)
  模型 C — LSTM               (原始时序 100×9)
  模型 D — Transformer        (原始时序 100×9)

  公平性说明：Ridge/RF 与 LSTM/Transformer 现在输入**同一套前 100 循环的原始特征**，
  仅表示形式不同（展平为向量 vs 保留时序），避免「时序模型吃原始序列、表格模型吃手工
  聚合统计量」造成的信息量不对等。

  评估方式：5 折交叉验证，固定 seed=42，报告均值 ± 标准差。
  预测目标：cycle_life（真实 EOL 标签，见 battery_analysis.py / common.py）。
  特征层：Feature B（9 维）= 6 动态 + 3 静态策略参数。
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset

# 固定全局随机种子，并获取 DataLoader 使用的生成器（保证 LSTM/Transformer 可复现）
g = C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()

# ============================================================
# 1. 读取数据（内部做电池顺序校验）
# ============================================================
FEATURES, y_life, meta = C.load_all_features(with_dqdv=False)   # 只需 Feature B，跳过 dQ/dV 读取
X_seq = FEATURES['B']                        # (124, 100, 9) Feature B
X_flat = C.flatten_sequence(X_seq)           # (124, 900) 展平，公平地给 Ridge/RF

N_BAT, SEQ_LEN, N_FEAT = X_seq.shape
print(f'数据: {N_BAT} 电池, 序列 {X_seq.shape}, 展平 {X_flat.shape}, 标签 {y_life.shape}')
print(f'cycle_life 范围: {y_life.min():.0f} - {y_life.max():.0f}, 均值: {y_life.mean():.1f}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device}')

kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)

def eval_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
    }


def summarize(metrics):
    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}


results = {}

# ============================================================
# 2. 模型 A: Ridge（展平原始序列，公平对比）
# ============================================================
print('\n' + '=' * 60)
print('模型 A: Ridge 回归 (展平原始序列, 900 维)')
print('=' * 60)
m_a = {'MAE': [], 'RMSE': [], 'R2': []}
for fold, (tr, va) in enumerate(kf.split(X_flat)):
    X_tr, X_val = X_flat[tr], X_flat[va]
    y_tr, y_val = y_life[tr], y_life[va]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    model = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=3, scoring='r2')
    model.fit(X_tr_s, y_tr)
    m = eval_metrics(y_val, model.predict(X_val_s))
    for k in m: m_a[k].append(m[k])
    print(f'  Fold {fold+1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R²={m["R2"]:.3f}')
results['Ridge'] = summarize(m_a)
print(f'  平均: R²={results["Ridge"]["R2"]["mean"]:.3f}±{results["Ridge"]["R2"]["std"]:.3f}')

# ============================================================
# 3. 模型 B: Random Forest（展平原始序列，公平对比）
# ============================================================
print('\n' + '=' * 60)
print('模型 B: Random Forest (展平原始序列, 900 维)')
print('=' * 60)
m_b = {'MAE': [], 'RMSE': [], 'R2': []}
for fold, (tr, va) in enumerate(kf.split(X_flat)):
    X_tr, X_val = X_flat[tr], X_flat[va]
    y_tr, y_val = y_life[tr], y_life[va]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                  random_state=C.SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    m = eval_metrics(y_val, model.predict(X_val))
    for k in m: m_b[k].append(m[k])
    print(f'  Fold {fold+1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R²={m["R2"]:.3f}')
results['RF'] = summarize(m_b)
print(f'  平均: R²={results["RF"]["R2"]["mean"]:.3f}±{results["RF"]["R2"]["std"]:.3f}')

# ============================================================
# 4. 训练辅助（序列模型共用）
# ============================================================
def make_loaders(X_tr, y_tr, X_val, y_val, batch_size=16):
    X_tr_s, X_val_s = C.standardize_sequences(X_tr, X_val)
    y_mean, y_std = y_tr.mean(), y_tr.std()
    y_tr_s = (y_tr - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std
    tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr_s))
    va_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val_s))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, generator=g)
    va_dl = DataLoader(va_ds, batch_size=64)
    return tr_dl, va_dl, y_mean, y_std


def train_and_eval(ModelClass, X_seq_, y_life_, lr=1e-3, epochs=200, name=''):
    m = {'MAE': [], 'RMSE': [], 'R2': []}
    for fold, (tr, va) in enumerate(kf.split(X_seq_)):
        X_tr, X_val = X_seq_[tr], X_seq_[va]
        y_tr, y_val = y_life_[tr], y_life_[va]
        tr_dl, va_dl, y_mean, y_std = make_loaders(X_tr, y_tr, X_val, y_val)
        model = ModelClass(input_dim=X_tr.shape[2])
        model = C.train_model(model, tr_dl, va_dl, epochs=epochs, lr=lr, device=device)
        model.eval()
        X_val_s = (X_val - X_tr.mean(axis=(0, 1))) / np.where(X_tr.std(axis=(0, 1)) == 0, 1.0, X_tr.std(axis=(0, 1)))
        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
        y_pred = y_pred_s * y_std + y_mean
        m_ = eval_metrics(y_val, y_pred)
        for k in m_: m[k].append(m_[k])
        print(f'  Fold {fold+1}: MAE={m_["MAE"]:.0f}, RMSE={m_["RMSE"]:.0f}, R²={m_["R2"]:.3f}')
    results[name] = summarize(m)
    print(f'  平均: R²={results[name]["R2"]["mean"]:.3f}±{results[name]["R2"]["std"]:.3f}')


# ============================================================
# 5. 模型 C: LSTM
# ============================================================
print('\n' + '=' * 60)
print('模型 C: LSTM (原始时序 100×9)')
print('=' * 60)
train_and_eval(LSTMPredictor, X_seq, y_life, lr=1e-3, name='LSTM')

# ============================================================
# 6. 模型 D: Transformer
# ============================================================
print('\n' + '=' * 60)
print('模型 D: Transformer (原始时序 100×9)')
print('=' * 60)
train_and_eval(TransformerPredictor, X_seq, y_life, lr=5e-4, name='Transformer')

# ============================================================
# 7. 汇总 + 落盘
# ============================================================
print('\n' + '=' * 60)
print('模型对比汇总（5 折交叉验证均值 ± 标准差）')
print('=' * 60)
print(f'{"模型":<15} {"MAE":>14} {"RMSE":>14} {"R²":>16}')
print('-' * 60)
for name in ['Ridge', 'RF', 'LSTM', 'Transformer']:
    r = results[name]
    print(f'{name:<15} {r["MAE"]["mean"]:>7.0f}±{r["MAE"]["std"]:.0f}  '
          f'{r["RMSE"]["mean"]:>7.0f}±{r["RMSE"]["std"]:.0f}  '
          f'{r["R2"]["mean"]:>8.3f}±{r["R2"]["std"]:.3f}')

C.save_metrics('model_comparison', {'input': 'Feature B (100×9), 展平 900 维给 Ridge/RF',
                                     'results': results})
print('\n全部完成!')
