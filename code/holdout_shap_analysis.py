# -*- coding: utf-8 -*-
"""
=============================================================================
留出测试集评估 — 多次重复分层划分（Transformer, Feature D）
=============================================================================
  评估协议（修正原先「单次 12 电池、硬编码 0.878」）：
    - 按 cycle_life 分位数分层，每层约取 10% test、10% val、80% train。
    - 重复 N_REPEATS 次不同随机划分，报告 R² / MAE / RMSE 的均值 ± 标准差。
    - 训练用 train，早停用 val，最终在 test 上评估（模型为早停最优权重）。

  说明：可解释性分析已移至 explain_transformer.py（对 Transformer 本身做
  Integrated Gradients / 注意力 rollout / 置换重要性），本脚本不再用随机森林
  的 SHAP 冒充主模型解释。
=============================================================================
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import common as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset

g = C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()

N_REPEATS = 10

# ============================================================
# 1. 读取 Feature D 数据
# ============================================================
FEATURES, y_life, meta = C.load_all_features()
X_seq = FEATURES['D']                     # (124, 100, 18)
print(f'数据: {X_seq.shape}, cycle_life 范围 {y_life.min():.0f}-{y_life.max():.0f}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def stratified_split(y, test_frac=0.1, val_frac=0.1, seed=0):
    """按寿命分位数分层，返回 (train_idx, val_idx, test_idx)。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    bins = np.digitize(y, np.percentile(y, [20, 40, 60, 80]))
    test_idx, val_idx, train_idx = [], [], []
    for b in range(6):
        in_bin = idx[bins[idx] == b]
        n = len(in_bin)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_idx += list(in_bin[:n_test])
        val_idx += list(in_bin[n_test:n_test + n_val])
        train_idx += list(in_bin[n_test + n_val:])
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


def run_one_split(tr, va, te):
    """训练一次并返回 test 集上的 (R2, MAE, RMSE)。"""
    X_tr, y_tr = X_seq[tr], y_life[tr]
    X_val, y_val = X_seq[va], y_life[va]
    X_te, y_te = X_seq[te], y_life[te]

    X_tr_s, X_val_s = C.standardize_sequences(X_tr, X_val)
    X_te_s = C.standardize_sequences(X_tr, X_te)[1]
    y_mean, y_std = y_tr.mean(), y_tr.std()
    tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor((y_tr - y_mean) / y_std))
    va_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, generator=g)
    va_dl = DataLoader(va_ds, batch_size=64)

    model = TransformerPredictor(input_dim=X_tr.shape[2])
    model = C.train_model(model, tr_dl, va_dl, epochs=200, lr=5e-4, device=device)
    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.FloatTensor(X_te_s).to(device)).cpu().numpy()
    y_pred = y_pred_s * y_std + y_mean
    return (float(r2_score(y_te, y_pred)),
            float(mean_absolute_error(y_te, y_pred)),
            float(np.sqrt(mean_squared_error(y_te, y_pred))))


# ============================================================
# 2. 多次重复评估
# ============================================================
print('\n' + '=' * 60)
print(f'留出测试集评估（{N_REPEATS} 次分层划分）')
print('=' * 60)

all_r2, all_mae, all_rmse = [], [], []
for rep in range(N_REPEATS):
    tr, va, te = stratified_split(y_life, seed=C.SEED + rep)
    r2, mae, rmse = run_one_split(tr, va, te)
    all_r2.append(r2); all_mae.append(mae); all_rmse.append(rmse)
    print(f'  repeat {rep+1:2d}: train={len(tr):3d} val={len(va):3d} test={len(te):3d}  '
          f'R²={r2:.3f}  MAE={mae:.0f}  RMSE={rmse:.0f}')

res = {
    'protocol': '分层留出 (10% val / 10% test / 80% train) × 重复次数',
    'n_repeats': N_REPEATS,
    'n_test_mean': float(np.mean([len(stratified_split(y_life, seed=C.SEED + r)[2]) for r in range(N_REPEATS)])),
    'R2': {'mean': float(np.mean(all_r2)), 'std': float(np.std(all_r2))},
    'MAE': {'mean': float(np.mean(all_mae)), 'std': float(np.std(all_mae))},
    'RMSE': {'mean': float(np.mean(all_rmse)), 'std': float(np.std(all_rmse))},
}

print('\n' + '=' * 60)
print('汇总（均值 ± 标准差）')
print('=' * 60)
print(f'  Test R²   = {res["R2"]["mean"]:.3f} ± {res["R2"]["std"]:.3f}')
print(f'  Test MAE  = {res["MAE"]["mean"]:.0f} ± {res["MAE"]["std"]:.0f} cycles')
print(f'  Test RMSE = {res["RMSE"]["mean"]:.0f} ± {res["RMSE"]["std"]:.0f} cycles')

C.save_metrics('holdout', res)
print('\n全部完成!')
