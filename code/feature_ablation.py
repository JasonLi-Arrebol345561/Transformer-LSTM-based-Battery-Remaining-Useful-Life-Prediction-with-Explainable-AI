# -*- coding: utf-8 -*-
"""
第二阶段: Feature Ablation — Transformer 逐层叠加
Feature A (基础退化) → B (+策略) → C (+SOC暴露) → D (+dQ/dV)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import h5py
from scipy.signal import find_peaks
from scipy.stats import skew
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 读取数据
# ============================================================
MAT_ORIG = r'E:\VSCode\BatteryLife-AI\MIT-Stanford Battery Dataset_cleaned.mat'
MAT_PROC = r'E:\VSCode\BatteryLife-AI\battery_analysis_results.mat'

print('读取数据...')
f_orig = h5py.File(MAT_ORIG, 'r')
f_proc = h5py.File(MAT_PROC, 'r')

bc = f_orig['batch_combined']
bi = f_proc['battery_info']
bats = sorted(bi.keys())

N_BAT = 124
SEQ_LEN = 100

# -------- 基础动态特征 (Feature A) --------
# SOH, IR, Tavg, QCharge, QDischarge, charge_time (6 dims)
FEAT_A = np.zeros((N_BAT, SEQ_LEN, 6), dtype=np.float32)

# -------- 策略特征 (Feature B 新增: C1, Q1, C2) --------
C1s = np.zeros(N_BAT); Q1s = np.zeros(N_BAT); C2s = np.zeros(N_BAT)

# -------- SOC暴露 (Feature C 新增: 4 bins) --------
SOC_BINS = [(0, 20), (20, 40), (40, 60), (60, 80)]
crate_bins = np.zeros((N_BAT, 4), dtype=np.float32)  # 静态

# -------- dQ/dV 统计量 (Feature D 新增: 5 stats per cycle) --------
FEAT_D = np.zeros((N_BAT, SEQ_LEN, 5), dtype=np.float32)

y_life = np.zeros(N_BAT, dtype=np.float32)

for idx, g in enumerate(bats):
    bg = bi[g]
    cl = int(bg['cycle_life'][()])
    y_life[idx] = cl

    # --- 动态特征 ---
    soh = bg['SOH'][()].flatten()
    ir  = bg['IR'][()].flatten()
    qc  = bg['QCharge'][()].flatten()
    qd  = bg['QDischarge'][()].flatten()
    ct  = bg['chargetime'][()].flatten()

    ref_s = bc['summary'][idx, 0]
    summ  = f_orig[ref_s]
    tavg  = summ['Tavg'][0, :]

    n_take = min(SEQ_LEN, len(soh))
    for t in range(n_take):
        FEAT_A[idx, t, 0] = soh[t]
        FEAT_A[idx, t, 1] = ir[t]
        FEAT_A[idx, t, 2] = tavg[t]
        FEAT_A[idx, t, 3] = qc[t]
        FEAT_A[idx, t, 4] = qd[t]
        FEAT_A[idx, t, 5] = ct[t]

    # --- 静态策略参数 ---
    c1 = float(bg['C1'][()])
    q1 = float(bg['Q1'][()])
    c2 = float(bg['C2'][()])
    C1s[idx] = c1 if c1 > 0 else 0
    Q1s[idx] = q1 if q1 > 0 else 0
    C2s[idx] = c2 if c2 > 0 else 0

    # --- SOC bin C-rate ---
    for j, (lo, hi) in enumerate(SOC_BINS):
        crates_in = []
        s1_lo, s1_hi = max(lo, 0), min(hi, Q1s[idx])
        if s1_hi > s1_lo: crates_in.append(C1s[idx])
        s2_lo, s2_hi = max(lo, Q1s[idx]), min(hi, 80)
        if s2_hi > s2_lo: crates_in.append(C2s[idx])
        s3_lo, s3_hi = max(lo, 80), min(hi, 100)
        if s3_hi > s3_lo: crates_in.append(1.0)
        crate_bins[idx, j] = np.mean(crates_in) if crates_in else 1.0

    # --- dQ/dV 统计量 (前 SEQ_LEN 个循环) ---
    ref_cycles = bc['cycles'][idx, 0]
    cycles_grp = f_orig[ref_cycles]
    dqdv_refs = cycles_grp['discharge_dQdV']

    for t in range(min(SEQ_LEN, dqdv_refs.shape[0])):
        dqdv = f_orig[dqdv_refs[t, 0]][()].flatten()
        if len(dqdv) < 10:
            # placeholder (first cycle often empty)
            FEAT_D[idx, t, :] = 0
            continue
        FEAT_D[idx, t, 0] = np.var(dqdv)
        peaks, props = find_peaks(-dqdv, prominence=0.01)  # dQ/dV 谷=放电峰
        FEAT_D[idx, t, 1] = len(peaks)
        if len(peaks) > 0:
            peak_heights = dqdv[peaks]
            main_idx = np.argmax(np.abs(peak_heights))
            FEAT_D[idx, t, 2] = peak_heights[main_idx]
            FEAT_D[idx, t, 3] = peaks[main_idx] / 1000.0
        FEAT_D[idx, t, 4] = skew(dqdv)

f_orig.close()
f_proc.close()
print(f'  读取完成: {N_BAT} 电池, 输入长度 {SEQ_LEN}')

# ============================================================
# 2. 构建四个特征集
# ============================================================
# Feature A: (124, 100, 6)
# Feature B: (124, 100, 9) = A + C1/Q1/C2 每步重复
# Feature C: (124, 100, 13) = B + SOC bin 每步重复
# Feature D: (124, 100, 18) = C + dQdV stats

def build_static_seq(vals, N, T):
    """将 (N,) 静态特征扩展为 (N, T, 1)"""
    return vals[:, np.newaxis, np.newaxis].repeat(T, axis=1)

FEAT_B = np.concatenate([FEAT_A,
                          build_static_seq(C1s, N_BAT, SEQ_LEN),
                          build_static_seq(Q1s, N_BAT, SEQ_LEN),
                          build_static_seq(C2s, N_BAT, SEQ_LEN)], axis=2)

FEAT_C = np.concatenate([FEAT_B,
                          build_static_seq(crate_bins[:, 0], N_BAT, SEQ_LEN),
                          build_static_seq(crate_bins[:, 1], N_BAT, SEQ_LEN),
                          build_static_seq(crate_bins[:, 2], N_BAT, SEQ_LEN),
                          build_static_seq(crate_bins[:, 3], N_BAT, SEQ_LEN)], axis=2)

FEAT_D_all = np.concatenate([FEAT_C, FEAT_D], axis=2)

FEATURES = {
    'A (基础退化)': FEAT_A,
    'B (+策略)':    FEAT_B,
    'C (+SOC暴露)': FEAT_C,
    'D (+dQ/dV)':   FEAT_D_all,
}

# ============================================================
# 3. Transformer 模型
# ============================================================
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
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device}')

def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3):
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

# ============================================================
# 4. 消融实验 (固定 5 折)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(np.arange(N_BAT)))  # 固定划分

all_results = {}

for feat_name, X in FEATURES.items():
    in_dim = X.shape[2]
    print(f'\n{"="*60}')
    print(f'Feature {feat_name}  (dim={in_dim})')
    print(f'{"="*60}')

    metrics = {'MAE': [], 'RMSE': [], 'R2': []}

    for fold, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_life[tr_idx], y_life[val_idx]

        # 标准化
        x_mean = X_tr.mean(axis=(0, 1), keepdims=True)
        x_std  = X_tr.std(axis=(0, 1), keepdims=True)
        x_std[x_std == 0] = 1.0
        X_tr_s  = (X_tr - x_mean) / x_std
        X_val_s = (X_val - x_mean) / x_std
        y_mean, y_std = y_tr.mean(), y_tr.std()
        y_tr_s = (y_tr - y_mean) / y_std

        train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr_s))
        val_ds   = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size=64)

        model = TransformerPredictor(input_dim=in_dim)
        model = train_model(model, train_dl, val_dl, epochs=200)

        model.eval()
        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(X_val_s).to(device)).cpu().numpy()
        y_pred = y_pred_s * y_std + y_mean

        mae  = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2   = r2_score(y_val, y_pred)
        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)
        metrics['R2'].append(r2)
        print(f'  Fold {fold+1}: MAE={mae:.0f}, RMSE={rmse:.0f}, R2={r2:.3f}')

    avg_mae  = np.mean(metrics['MAE'])
    avg_rmse = np.mean(metrics['RMSE'])
    avg_r2   = np.mean(metrics['R2'])
    std_r2   = np.std(metrics['R2'])
    all_results[feat_name] = (avg_mae, avg_rmse, avg_r2, std_r2, metrics['R2'])
    print(f'  平均: MAE={avg_mae:.0f}, RMSE={avg_rmse:.0f}, R2={avg_r2:.3f}±{std_r2:.3f}')

# ============================================================
# 5. 汇总 + 图表
# ============================================================
print('\n' + '='*60)
print('消融实验结果汇总')
print('='*60)

stages = list(FEATURES.keys())
r2_values = [all_results[s][2] for s in stages]
r2_stds   = [all_results[s][3] for s in stages]
r2_deltas = [r2_values[0]] + [r2_values[i] - r2_values[i-1] for i in range(1, len(stages))]

print(f'\n{"阶段":<20} {"维度":<6} {"R²":<16} {"ΔR²":<12}')
print('-' * 54)
for i, s in enumerate(stages):
    print(f'{s:<20} {FEATURES[s].shape[2]:<6} {r2_values[i]:.3f} ± {r2_stds[i]:.3f}    +{r2_deltas[i]:.3f}')

# 柱状图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

FIG_DIR = r'E:\VSCode\BatteryLife-AI\figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(9, 5.5))
short_names = ['A\n基础退化', 'B\n+策略', 'C\n+SOC暴露', 'D\n+dQ/dV']
bars = ax.bar(short_names, r2_values, color=['#95A5A6', '#2E86AB', '#4ECDC4', '#FF6B6B'],
              edgecolor='white', linewidth=1.5, width=0.55)

# R2 标注
for bar, r2, std in zip(bars, r2_values, r2_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'R2={r2:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 增量箭头
for i in range(1, len(stages)):
    mid_x = (bars[i-1].get_x() + bars[i-1].get_width()/2 + bars[i].get_x() + bars[i].get_width()/2) / 2
    ax.annotate(f'+{r2_deltas[i]:.3f}',
                xy=(mid_x, max(r2_values[i-1], r2_values[i]) + 0.06),
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

ax.set_ylabel('R² 分数', fontsize=13)
ax.set_title('特征消融实验: Transformer 逐层特征叠加\n'
             'R² 提升揭示各物理变量对预测的贡献',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, max(r2_values) + 0.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'feature_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n图表: {FIG_DIR}/feature_ablation.png')
print('全部完成!')
