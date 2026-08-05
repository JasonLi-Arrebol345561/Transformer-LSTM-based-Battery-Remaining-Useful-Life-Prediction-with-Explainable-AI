# -*- coding: utf-8 -*-
"""
第一阶段：固定特征 × 四模型对比
Linear Regression / Random Forest / LSTM / Transformer
预测目标: cycle_life (从早期循环特征)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import h5py
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 读取数据
# ============================================================
MAT_ORIG = r'E:\VSCode\BatteryLife-AI\MIT-Stanford Battery Dataset_cleaned.mat'
MAT_PROC = r'E:\VSCode\BatteryLife-AI\battery_analysis_results.mat'

f_orig = h5py.File(MAT_ORIG, 'r')
f_proc = h5py.File(MAT_PROC, 'r')

bc = f_orig['batch_combined']
bi = f_proc['battery_info']
bats = sorted(bi.keys())

N_BAT = 124
SEQ_LEN = 100          # 使用前100个循环
N_DYNAMIC = 6          # 动态特征数: SOH, IR, Tavg, QCharge, QDischarge, charge_time
N_STATIC = 3           # 静态特征数: C1, Q1, C2
N_FEAT = N_DYNAMIC + N_STATIC  # 9

X_seq = np.zeros((N_BAT, SEQ_LEN, N_FEAT), dtype=np.float32)
y_life = np.zeros(N_BAT, dtype=np.float32)

for idx, g in enumerate(bats):
    bg = bi[g]
    cl = int(bg['cycle_life'][()])
    y_life[idx] = cl

    # 动态特征 (从 processed .mat)
    soh  = bg['SOH'][()].flatten()
    ir   = bg['IR'][()].flatten()
    qc   = bg['QCharge'][()].flatten()
    qd   = bg['QDischarge'][()].flatten()
    ct   = bg['chargetime'][()].flatten()

    # 温度 (从原始 .mat)
    ref_s = bc['summary'][idx, 0]
    summ  = f_orig[ref_s]
    tavg  = summ['Tavg'][0, :]

    # 取前 SEQ_LEN 个循环 (不足则取全部，但目前所有电池 ≥ 170 cycles)
    n_take = min(SEQ_LEN, len(soh))
    for t in range(n_take):
        X_seq[idx, t, 0] = soh[t]
        X_seq[idx, t, 1] = ir[t]
        X_seq[idx, t, 2] = tavg[t]
        X_seq[idx, t, 3] = qc[t]
        X_seq[idx, t, 4] = qd[t]
        X_seq[idx, t, 5] = ct[t]

    # 静态特征 (每循环重复)
    c1 = float(bg['C1'][()])
    q1 = float(bg['Q1'][()])
    c2 = float(bg['C2'][()])
    X_seq[idx, :n_take, 6] = c1 if c1 > 0 else 0
    X_seq[idx, :n_take, 7] = q1 if q1 > 0 else 0
    X_seq[idx, :n_take, 8] = c2 if c2 > 0 else 0

f_orig.close()
f_proc.close()

print(f'数据: {N_BAT} 电池, 输入形状 {X_seq.shape}, 输出形状 {y_life.shape}')
print(f'cycle_life 范围: {y_life.min():.0f} - {y_life.max():.0f}, 均值: {y_life.mean():.1f}')

# ============================================================
# 2. 特征聚合 (给 LR / RF 用)
# ============================================================
def aggregate_features(X_seq):
    """将 (N, T, D) 序列聚合为 (N, D*5) 固定长度向量
       每个动态特征 → [mean, std, min, max, trend_slope]
       静态特征 → 直接用均值"""
    N, T, D = X_seq.shape
    feats = []
    feat_names = []
    dyn_names = ['SOH', 'IR', 'Tavg', 'QCharge', 'QDischarge', 'charge_time']
    sta_names = ['C1', 'Q1', 'C2']

    # 动态特征聚合
    for d in range(N_DYNAMIC):
        vals = X_seq[:, :, d]
        f_mean = np.mean(vals, axis=1)
        f_std  = np.std(vals, axis=1)
        f_min  = np.min(vals, axis=1)
        f_max  = np.max(vals, axis=1)
        # 线性趋势斜率
        x = np.arange(T)
        f_slope = np.array([np.polyfit(x, vals[i], 1)[0] for i in range(N)])
        feats.extend([f_mean, f_std, f_min, f_max, f_slope])
        for agg in ['mean', 'std', 'min', 'max', 'slope']:
            feat_names.append(f'{dyn_names[d]}_{agg}')

    # 静态特征 (取均值 = 原值)
    for d in range(N_DYNAMIC, N_FEAT):
        feats.append(np.mean(X_seq[:, :, d], axis=1))
        feat_names.append(sta_names[d - N_DYNAMIC])

    return np.column_stack(feats), feat_names

X_agg, agg_names = aggregate_features(X_seq)
print(f'聚合特征: {X_agg.shape[1]} 维')

# ============================================================
# 3. 5折交叉验证
# ============================================================
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def eval_metrics(y_true, y_pred):
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred),
    }

results = {}

# ============================================================
# Model A: Ridge Linear Regression
# ============================================================
print('\n' + '='*60)
print('Model A: Ridge 回归')
print('='*60)

metrics_a = {'MAE': [], 'RMSE': [], 'R2': []}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_agg)):
    X_tr, X_val = X_agg[train_idx], X_agg[val_idx]
    y_tr, y_val = y_life[train_idx], y_life[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=3, scoring='r2')
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_val_s)
    m = eval_metrics(y_val, y_pred)
    for k in m:
        metrics_a[k].append(m[k])
    print(f'  Fold {fold+1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R2={m["R2"]:.3f}')

results['Ridge'] = {k: (np.mean(v), np.std(v)) for k, v in metrics_a.items()}
print(f'  平均: MAE={np.mean(metrics_a["MAE"]):.0f}±{np.std(metrics_a["MAE"]):.0f}, '
      f'RMSE={np.mean(metrics_a["RMSE"]):.0f}±{np.std(metrics_a["RMSE"]):.0f}, '
      f'R2={np.mean(metrics_a["R2"]):.3f}±{np.std(metrics_a["R2"]):.3f}')

# ============================================================
# Model B: Random Forest
# ============================================================
print('\n' + '='*60)
print('Model B: Random Forest')
print('='*60)

metrics_b = {'MAE': [], 'RMSE': [], 'R2': []}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_agg)):
    X_tr, X_val = X_agg[train_idx], X_agg[val_idx]
    y_tr, y_val = y_life[train_idx], y_life[val_idx]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    m = eval_metrics(y_val, y_pred)
    for k in m:
        metrics_b[k].append(m[k])
    print(f'  Fold {fold+1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R2={m["R2"]:.3f}')

results['RF'] = {k: (np.mean(v), np.std(v)) for k, v in metrics_b.items()}
print(f'  平均: MAE={np.mean(metrics_b["MAE"]):.0f}±{np.std(metrics_b["MAE"]):.0f}, '
      f'RMSE={np.mean(metrics_b["RMSE"]):.0f}±{np.std(metrics_b["RMSE"]):.0f}, '
      f'R2={np.mean(metrics_b["R2"]):.3f}±{np.std(metrics_b["R2"]):.3f}')

# ============================================================
# Model C: LSTM
# ============================================================
print('\n' + '='*60)
print('Model C: LSTM')
print('='*60)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        scheduler.step(val_loss)

    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  设备: {device}')

metrics_c = {'MAE': [], 'RMSE': [], 'R2': []}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
    X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
    y_tr, y_val = y_life[train_idx], y_life[val_idx]

    # 标准化
    x_mean, x_std = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1))
    x_std[x_std == 0] = 1.0
    X_tr_s = (X_tr - x_mean) / x_std
    X_val_s = (X_val - x_mean) / x_std
    y_mean, y_std = y_tr.mean(), y_tr.std()
    y_tr_s = (y_tr - y_mean) / y_std

    train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr_s))
    val_ds   = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=64)

    model = LSTMPredictor()
    model = train_model(model, train_dl, val_dl, epochs=200, device=device)

    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
    y_pred = y_pred_s * y_std + y_mean
    m = eval_metrics(y_val, y_pred)
    for k in m:
        metrics_c[k].append(m[k])
    print(f'  Fold {fold+1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R2={m["R2"]:.3f}')

results['LSTM'] = {k: (np.mean(v), np.std(v)) for k, v in metrics_c.items()}
print(f'  平均: MAE={np.mean(metrics_c["MAE"]):.0f}±{np.std(metrics_c["MAE"]):.0f}, '
      f'RMSE={np.mean(metrics_c["RMSE"]):.0f}±{np.std(metrics_c["RMSE"]):.0f}, '
      f'R2={np.mean(metrics_c["R2"]):.3f}±{np.std(metrics_c["R2"]):.3f}')

# ============================================================
# Model D: Transformer
# ============================================================
print('\n' + '='*60)
print('Model D: Transformer')
print('='*60)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

metrics_d = {'MAE': [], 'RMSE': [], 'R2': []}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
    X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
    y_tr, y_val = y_life[train_idx], y_life[val_idx]

    x_mean, x_std = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1))
    x_std[x_std == 0] = 1.0
    X_tr_s = (X_tr - x_mean) / x_std
    X_val_s = (X_val - x_mean) / x_std
    y_mean, y_std = y_tr.mean(), y_tr.std()
    y_tr_s = (y_tr - y_mean) / y_std

    train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr_s))
    val_ds   = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=64)

    model = TransformerPredictor()
    model = train_model(model, train_dl, val_dl, epochs=200, lr=5e-4, device=device)

    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
    y_pred = y_pred_s * y_std + y_mean
    m = eval_metrics(y_val, y_pred)
    for k in m:
        metrics_d[k].append(m[k])
    print(f'  Fold {fold+1}: MAE={m["MAE"]:.0f}, RMSE={m["RMSE"]:.0f}, R2={m["R2"]:.3f}')

results['Transformer'] = {k: (np.mean(v), np.std(v)) for k, v in metrics_d.items()}
print(f'  平均: MAE={np.mean(metrics_d["MAE"]):.0f}±{np.std(metrics_d["MAE"]):.0f}, '
      f'RMSE={np.mean(metrics_d["RMSE"]):.0f}±{np.std(metrics_d["RMSE"]):.0f}, '
      f'R2={np.mean(metrics_d["R2"]):.3f}±{np.std(metrics_d["R2"]):.3f}')

# ============================================================
# 4. 汇总对比
# ============================================================
print('\n' + '='*60)
print('汇总对比')
print('='*60)
print(f'{"模型":<15} {"MAE":>10} {"RMSE":>10} {"R²":>10}')
print('-' * 45)
for name in ['Ridge', 'RF', 'LSTM', 'Transformer']:
    r = results[name]
    print(f'{name:<15} {r["MAE"][0]:>8.0f}±{r["MAE"][1]:.0f}  '
          f'{r["RMSE"][0]:>8.0f}±{r["RMSE"][1]:.0f}  '
          f'{r["R2"][0]:>8.3f}±{r["R2"][1]:.3f}')

print('\n全部完成!')
