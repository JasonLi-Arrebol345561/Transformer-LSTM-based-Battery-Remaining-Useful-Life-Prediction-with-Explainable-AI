# -*- coding: utf-8 -*-
"""
=============================================================================
Transformer 可解释性分析（对主模型本身，而非随机森林）
=============================================================================
  对「产生预测结论的模型」—— Transformer（Feature D, 18 维）——做三类归因：
    1. Integrated Gradients    ：输入 (100×18) 逐时间步×特征归因 → 特征重要性
    2. 置换重要性 (Permutation)：逐特征打乱，测测试集 R² 下降
    3. 注意力 rollout          ：逐层头平均 + 残差连乘，得时间步贡献矩阵

  基线对照（用于防止过度解读）：
    - 随机初始化模型：其 IG / rollout 应与训练模型明显不同（否则说明归因无信息）
    - 时间轴打乱对照：打乱 100 个时间步后重训，R² 明显下降 → 模型确实依赖时序结构

  输出：
    - figures/ig_importance.png        IG 特征重要性（柱状图）
    - figures/permutation_importance.png 置换重要性（柱状图）
    - figures/attention_rollout.png    rollout 注意力热图
    - figures/attention_profile.png    rollout 时间步贡献分布
=============================================================================
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import common as C
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

C.seed_all(C.SEED)
PositionalEncoding, TransformerPredictor, LSTMPredictor = C.build_models()

FEAT_NAMES = ['SOH', 'IR', 'Tavg', 'QCharge', 'QDischarge', 'charge_time',
              'C1', 'Q1', 'C2',
              'Crate_0-20', 'Crate_20-40', 'Crate_40-60', 'Crate_60-80',
              'dQdV_var', 'dQdV_npeaks', 'dQdV_peak_h', 'dQdV_peak_pos', 'dQdV_skew']

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 1. 数据 + 训练最终模型（分层留出）
# ============================================================
FEATURES, y_life, meta = C.load_all_features()
X_seq = FEATURES['D']                     # (124, 100, 18)
N, T, D = X_seq.shape


def stratified_split(y, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    bins = np.digitize(y, np.percentile(y, [20, 40, 60, 80]))
    tr, va, te = [], [], []
    for b in range(6):
        in_bin = idx[bins[idx] == b]
        n = len(in_bin)
        n_te = max(1, int(n * 0.1)); n_va = max(1, int(n * 0.1))
        te += list(in_bin[:n_te]); va += list(in_bin[n_te:n_te + n_va]); tr += list(in_bin[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


tr, va, te = stratified_split(y_life, seed=C.SEED)
X_tr, y_tr = X_seq[tr], y_life[tr]
X_val, y_val = X_seq[va], y_life[va]
X_te, y_te = X_seq[te], y_life[te]

X_tr_s, X_val_s = C.standardize_sequences(X_tr, X_val)
X_te_s = C.standardize_sequences(X_tr, X_te)[1]
x_mean, x_std = X_tr.mean(axis=(0, 1), keepdims=True), X_tr.std(axis=(0, 1), keepdims=True)
y_mean, y_std = y_tr.mean(), y_tr.std()

torch.manual_seed(C.SEED * 1000)
g = C.fold_generator(C.SEED, 0)
tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor((y_tr - y_mean) / y_std))
va_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, generator=g)
va_dl = DataLoader(va_ds, batch_size=64)

model = TransformerPredictor(input_dim=D)
model = C.train_model(model, tr_dl, va_dl, epochs=200, lr=5e-4, device=device)
model.eval()

with torch.no_grad():
    y_te_pred = model(torch.FloatTensor(X_te_s).to(device)).cpu().numpy() * y_std + y_mean
test_r2 = float(r2_score(y_te, y_te_pred))
print(f'训练完成，Test R² = {test_r2:.3f}（{len(te)} 个测试电池）')


def predict(model, X_s):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_s).to(device)).cpu().numpy() * y_std + y_mean


# ============================================================
# 2. Integrated Gradients
# ============================================================
def integrated_gradients(model, x, baseline, steps=50):
    model.eval()
    ig = torch.zeros_like(x)
    for k in range(steps):
        alpha = k / (steps - 1)
        xa = baseline + alpha * (x - baseline)
        xa = xa.clone().detach().requires_grad_(True)
        out = model(xa)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        out.sum().backward()
        ig += xa.grad
    return (x - baseline) * (ig / steps)


print('\n计算 Integrated Gradients ...')
X_te_t = torch.FloatTensor(X_te_s).to(device)
baseline = torch.zeros_like(X_te_t)
ig_all = []
for i in range(X_te_t.shape[0]):
    ig = integrated_gradients(model, X_te_t[i:i + 1], baseline[i:i + 1]).cpu().numpy()
    ig_all.append(np.abs(ig[0]))
ig_abs = np.array(ig_all)                    # (n_test, T, D)
ig_feat = ig_abs.mean(axis=(0, 1))           # 特征重要性：平均 |IG|
ig_time = ig_abs.mean(axis=(0, 2))           # 时间步贡献分布

# 随机初始化基线
rand_model = TransformerPredictor(input_dim=D).to(device)
ig_rand = []
for i in range(X_te_t.shape[0]):
    ig = integrated_gradients(rand_model, X_te_t[i:i + 1], baseline[i:i + 1]).cpu().numpy()
    ig_rand.append(np.abs(ig[0]))
ig_rand_feat = np.array(ig_rand).mean(axis=(0, 1))

fig, ax = plt.subplots(figsize=(10, 5.5))
order = np.argsort(ig_feat)
ax.barh([FEAT_NAMES[i] for i in order], ig_feat[order], color='#2E86AB')
ax.set_xlabel('平均 |IG| 归因强度')
ax.set_title('Transformer Integrated Gradients 特征重要性\n(对测试集平均)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'ig_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  {C.fig_dir()}/ig_importance.png')

# ============================================================
# 3. 置换重要性（逐特征打乱 → 测试 R² 下降）
# ============================================================
print('计算置换重要性（固定 seed 多次打乱取均值） ...')
perm_imp = np.zeros(D)
base_pred = predict(model, X_te_s)
base_r2 = r2_score(y_te, base_pred)
N_PERM = 10
for d in range(D):
    drops = []
    for r in range(N_PERM):
        rng = np.random.RandomState(C.SEED * 1000 + d * 100 + r)  # 固定 seed，可复现
        Xp = X_te_s.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, :, d] = X_te_s[perm, :, d]
        drops.append(base_r2 - r2_score(y_te, predict(model, Xp)))
    perm_imp[d] = np.mean(drops)

fig, ax = plt.subplots(figsize=(10, 5.5))
order = np.argsort(perm_imp)
ax.barh([FEAT_NAMES[i] for i in order], perm_imp[order], color='#4ECDC4')
ax.set_xlabel('ΔR²（打乱该特征后测试 R² 下降）')
ax.set_title('Transformer 置换重要性\n(下降越多越重要；负值=打乱反而更好，属噪声)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'permutation_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  {C.fig_dir()}/permutation_importance.png')

# ============================================================
# 4. 注意力 rollout
# ============================================================
print('计算注意力 rollout ...')
sample = X_te_t[:1]
rollout = C.attention_rollout(model, sample)      # (T, T)
rollout_rand = C.attention_rollout(rand_model, sample)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im0 = axes[0].imshow(rollout, cmap='YlOrRd', aspect='auto')
axes[0].set_title('训练模型 rollout 注意力')
axes[0].set_xlabel('Key 位置'); axes[0].set_ylabel('Query 位置')
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(rollout_rand, cmap='YlOrRd', aspect='auto')
axes[1].set_title('随机初始化模型 rollout（基线）')
axes[1].set_xlabel('Key 位置'); axes[1].set_ylabel('Query 位置')
fig.colorbar(im1, ax=axes[1])
fig.suptitle('注意力 rollout 对比（第 1 个测试电池）\n训练模型 vs 随机基线', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'attention_rollout.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  {C.fig_dir()}/attention_rollout.png')

# 时间步贡献分布（rollout 最后一行的归一化 = 最终预测对历史各步的依赖）
fig, ax = plt.subplots(figsize=(8, 5))
last_row = rollout[-1]
last_row = last_row / (last_row.sum() + 1e-12)
ax.plot(last_row, label='训练模型', linewidth=2)
last_row_r = rollout_rand[-1]
last_row_r = last_row_r / (last_row_r.sum() + 1e-12)
ax.plot(last_row_r, label='随机基线', linewidth=2, linestyle='--')
ax.set_xlabel('Key 位置 (前 100 循环中的周期)')
ax.set_ylabel('归一化贡献')
ax.set_title('rollout 时间步贡献分布：最终预测依赖历史哪些步')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'attention_profile.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  {C.fig_dir()}/attention_profile.png')

# ============================================================
# 5. 时间轴打乱对照
# ============================================================
print('\n时间轴打乱对照 ...')
rng = np.random.RandomState(0)
X_shuf = X_seq.copy()
for i in range(N):
    p = rng.permutation(T)
    X_shuf[i] = X_seq[i][p]

X_shuf_tr_s, X_shuf_val_s = C.standardize_sequences(X_shuf[tr], X_shuf[va])
torch.manual_seed(C.SEED * 1000 + 1)
g2 = C.fold_generator(C.SEED, 1)
tr_ds2 = TensorDataset(torch.FloatTensor(X_shuf_tr_s), torch.FloatTensor((y_tr - y_mean) / y_std))
va_ds2 = TensorDataset(torch.FloatTensor(X_shuf_val_s), torch.FloatTensor((y_val - y_mean) / y_std))
tr_dl2 = DataLoader(tr_ds2, batch_size=16, shuffle=True, generator=g2)
va_dl2 = DataLoader(va_ds2, batch_size=64)
model_shuf = TransformerPredictor(input_dim=D)
model_shuf = C.train_model(model_shuf, tr_dl2, va_dl2, epochs=200, lr=5e-4, device=device)
X_shuf_te_s = C.standardize_sequences(X_shuf[tr], X_shuf[te])[1]
with torch.no_grad():
    y_shuf_pred = model_shuf(torch.FloatTensor(X_shuf_te_s).to(device)).cpu().numpy() * y_std + y_mean
shuf_r2 = float(r2_score(y_te, y_shuf_pred))
print(f'  原始时序 Test R² = {test_r2:.3f}')
print(f'  时间打乱 Test R² = {shuf_r2:.3f}  (下降说明模型依赖时序结构)')

# ============================================================
# 6. 落盘
# ============================================================
ig_order = [FEAT_NAMES[i] for i in np.argsort(ig_feat)[::-1]]
perm_order = [FEAT_NAMES[i] for i in np.argsort(perm_imp)[::-1]]
C.save_metrics('explain', {
    'test_r2': test_r2,
    'time_shuffle_test_r2': shuf_r2,
    'ig_feature_importance': {FEAT_NAMES[i]: float(ig_feat[i]) for i in range(D)},
    'permutation_importance': {FEAT_NAMES[i]: float(perm_imp[i]) for i in range(D)},
    'ig_top5': ig_order[:5],
    'permutation_top5': perm_order[:5],
})
print('\n全部完成!')
