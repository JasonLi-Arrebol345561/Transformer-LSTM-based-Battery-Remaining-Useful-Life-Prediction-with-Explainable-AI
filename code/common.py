# -*- coding: utf-8 -*-
"""
=============================================================================
共享模块 — 路径 / 数据加载(含顺序校验) / 特征构建 / 模型 / 训练 / metrics
=============================================================================
  本模块是下游所有脚本（model_comparison / feature_ablation /
  holdout_eval / final_figures / explain_transformer）的单一正确来源，
  消除原先三处重复的特征读取逻辑，并把关键修正（顺序校验、宽度加权 SOC C-rate、
  固定随机种子、真实 cycle_life 标签）统一收敛到这里。

  使用约定：
    - 目标变量 cycle_life 已改为「真实 EOL 标签」（bc['cycle_life']），不再是
      QDischarge 数组长度（n_rows）的代理值。该修正在 battery_analysis.py 落盘。
    - 读端必须调用 assert_battery_order() 校验派生文件与原始 batch_combined 的顺序，
      防止 Tavg / dQ/dV 被静默接到错误电池上。
=============================================================================
"""

import os
import re
import json
import random
import numpy as np
import h5py
from scipy.signal import find_peaks
from scipy.stats import skew

# ============================================================
# 常量
# ============================================================
N_BAT = 124
SEQ_LEN = 100
NOMINAL_CAPACITY = 1.1          # A123 LFP 电芯额定容量 (Ah)

# 模型特征用的 4 个 SOC 区间（上限 80%，因 80-100% 恒为 1C CC-CV，不作为区分特征）
SOC_BINS = [(0, 20), (20, 40), (40, 60), (60, 80)]
# 统计/相关性分析用的 5 个 SOC 区间（含 80-100%）
SOC_BINS_ALL = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

DYNAMIC_NAMES = ['SOH', 'IR', 'Tavg', 'QCharge', 'QDischarge', 'charge_time']
STATIC_NAMES = ['C1', 'Q1', 'C2']

SEED = 42

# ============================================================
# 路径
# ============================================================
def proj_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mat_orig_path():
    return os.path.join(proj_dir(), 'data', 'MIT-Stanford Battery Dataset_cleaned.mat')


def mat_proc_path():
    return os.path.join(proj_dir(), 'data', 'battery_analysis_results.mat')


def fig_dir():
    return os.path.join(proj_dir(), 'figures')


def results_dir():
    return os.path.join(proj_dir(), 'results')


# ============================================================
# 策略解析
# ============================================================
def decode_policy(f, ref):
    """将 HDF5 中 policy_readable 的引用解码为字符串，如 "3.6C(80%)-4.6C"。"""
    obj = f[ref]
    chars = [chr(int(x[0])) for x in obj[:]]
    return ''.join(chars)


def parse_policy(policy_str):
    """从策略字符串解析 (C1, Q1, C2)；失败返回 (None, None, None)。"""
    s = policy_str.replace('-newstructure', '')
    m = re.match(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C', s)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


# ============================================================
# 随机种子（含 torch，保证 LSTM/Transformer 可复现）
# ============================================================
def seed_all(seed=SEED):
    """设置全局随机种子（random/numpy/torch）。不返回共享 generator（见 fold_generator）。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fold_generator(seed=SEED, fold=0, tag=0):
    """
    为单个 (fold, tag) 训练返回一个独立、可复现的 DataLoader generator。

    修复：原先 seed_all 返回一个全局共享的 torch.Generator，被前一个模型/折
    持续消耗后，后一个模型/折的 DataLoader 洗牌序列会漂移，导致同一配置在不同
    脚本里结果不一致（如 Transformer+B 在 model_comparison 得 0.827、在
    feature_ablation 得 0.870）。改为每折独立、由 (seed, fold, tag) 确定的种子。
    """
    import torch
    return torch.Generator().manual_seed(seed * 1000 + fold * 100 + tag)


def mape(y_true, y_pred):
    """平均绝对百分比误差（%）。y_true 不应含 0。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# ============================================================
# 数据加载 + 顺序校验
# ============================================================
def open_data():
    """打开原始与预处理两个 .mat，返回 (f_orig, f_proc, bc, bi, bats)。"""
    f_orig = h5py.File(mat_orig_path(), 'r')
    f_proc = h5py.File(mat_proc_path(), 'r')
    bc = f_orig['batch_combined']
    bi = f_proc['battery_info']
    bats = sorted(bi.keys())          # bat_001 .. bat_124
    return f_orig, f_proc, bc, bi, bats


def close_data(f_orig, f_proc):
    f_orig.close()
    f_proc.close()


def assert_battery_order(f_orig, bc, bi, bats):
    """
    校验派生文件的电池顺序与原始 batch_combined 一致。

    用两个独立标识做交叉核对：
      1) 策略字符串：派生 policy == 原始 policy_readable 解码结果（强校验，唯一标识电池）
      2) 真实寿命：派生 cycle_life == 原始 bc['cycle_life']（附加校验）

    若不一致立即抛错，避免 Tavg / dQ/dV 被静默接到错误电池。
    """
    n_mismatch = 0
    for idx, g in enumerate(bats):
        bg = bi[g]
        derived_policy = bg['policy'][()].decode('utf-8')
        orig_policy = decode_policy(f_orig, bc['policy_readable'][idx, 0])
        derived_life = float(bg['cycle_life'][()])
        orig_life = float(f_orig[bc['cycle_life'][idx, 0]][0, 0])
        if derived_policy != orig_policy or not np.isclose(derived_life, orig_life):
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f'  [MISMATCH] idx={idx} {g}: policy "{derived_policy}" vs "{orig_policy}", '
                      f'life {derived_life} vs {orig_life}')
    if n_mismatch:
        raise RuntimeError(f'电池顺序校验失败: {n_mismatch} 处不匹配（派生文件与原始数据顺序不一致）')
    print(f'  OK 顺序校验通过（{len(bats)} 个电池，策略 + 真实寿命 双重核对）')


# ============================================================
# SOC 区间等效 C-rate（宽度加权）
# ============================================================
def soc_bin_crates(c1, q1, c2, bins=SOC_BINS):
    """
    计算某电池在每个 SOC bin 内的等效 C-rate，按「SOC 覆盖宽度」加权平均。

    充电三段模型：阶段1 [0, q1] 用 C1；阶段2 [q1, 80] 用 C2；阶段3 [80,100] 用 1C(CC-CV)。
    对每个 bin，用各阶段与该 bin 的交叠 SOC 宽度作为权重求加权平均（修正原先的简单均值）。
    无交叠的 bin 返回 NaN。
    """
    crates = np.full(len(bins), np.nan)
    stages = [(0.0, q1, c1), (q1, 80.0, c2), (80.0, 100.0, 1.0)]
    for j, (lo, hi) in enumerate(bins):
        w_sum = 0.0
        w_val = 0.0
        for s_lo, s_hi, rate in stages:
            ov_lo = max(lo, s_lo)
            ov_hi = min(hi, s_hi)
            if ov_hi > ov_lo:
                w = ov_hi - ov_lo
                w_sum += w
                w_val += w * rate
        if w_sum > 0:
            crates[j] = w_val / w_sum
    return crates


def build_static_seq(vals, T):
    """把 (N,) 静态特征扩展为 (N, T, 1)，便于与动态序列在时间维拼接。"""
    return vals[:, np.newaxis, np.newaxis].repeat(T, axis=1)


# ============================================================
# 特征构建（一次读取，构建 Feature A/B/C/D）
# ============================================================
def load_all_features(seq_len=SEQ_LEN, with_dqdv=True):
    """
    一次读取原始 + 预处理数据，构建四层特征并返回。

    返回:
        FEATURES : dict，键 'A'/'B'/'C'/'D'
            A (N,T,6)  动态退化信号 SOH/IR/Tavg/QCharge/QDischarge/charge_time
            B (N,T,9)  A + 静态 C1/Q1/C2
            C (N,T,13) B + 4 个 SOC 区间等效 C-rate（宽度加权）
            D (N,T,18) C + 5 个 dQ/dV 统计量
        y_life   : (N,) 真实 cycle_life 标签
        meta     : list[dict]，每电池 {C1,Q1,C2,cycle_life,policy}

    参数:
        with_dqdv : 是否需要 dQ/dV 特征（False 可跳过昂贵的 dQ/dV 读取，省 I/O）。
    """
    f_orig, f_proc, bc, bi, bats = open_data()
    assert_battery_order(f_orig, bc, bi, bats)

    N = N_BAT
    T = seq_len

    FEAT_A = np.zeros((N, T, 6), dtype=np.float32)
    C1s = np.zeros(N)
    Q1s = np.zeros(N)
    C2s = np.zeros(N)
    crate_bins = np.zeros((N, 4), dtype=np.float32)
    FEAT_D = np.zeros((N, T, 5), dtype=np.float32)
    y_life = np.zeros(N, dtype=np.float32)
    meta = []

    for idx, g in enumerate(bats):
        bg = bi[g]
        cl = float(bg['cycle_life'][()])          # 真实 EOL 标签
        y_life[idx] = cl

        soh = bg['SOH'][()].flatten()
        ir = bg['IR'][()].flatten()
        qc = bg['QCharge'][()].flatten()
        qd = bg['QDischarge'][()].flatten()
        ct = bg['chargetime'][()].flatten()

        ref_s = bc['summary'][idx, 0]
        summ = f_orig[ref_s]
        tavg = summ['Tavg'][0, :]

        c1 = float(bg['C1'][()]); c1 = c1 if c1 > 0 else 0
        q1 = float(bg['Q1'][()]); q1 = q1 if q1 > 0 else 0
        c2 = float(bg['C2'][()]); c2 = c2 if c2 > 0 else 0
        C1s[idx] = c1; Q1s[idx] = q1; C2s[idx] = c2
        meta.append({'C1': c1, 'Q1': q1, 'C2': c2, 'cycle_life': cl,
                     'policy': bg['policy'][()].decode('utf-8')})

        n_take = min(T, len(soh))
        for t in range(n_take):
            FEAT_A[idx, t, 0] = soh[t]
            FEAT_A[idx, t, 1] = ir[t]
            FEAT_A[idx, t, 2] = tavg[t]
            FEAT_A[idx, t, 3] = qc[t]
            FEAT_A[idx, t, 4] = qd[t]
            FEAT_A[idx, t, 5] = ct[t]

        # SOC 区间 C-rate（宽度加权，4 bins）
        crate_bins[idx] = soc_bin_crates(c1, q1, c2, SOC_BINS)

        # dQ/dV 统计量（5 维）—— 仅在需要时读取（省 I/O）
        if with_dqdv:
            ref_c = bc['cycles'][idx, 0]
            cg = f_orig[ref_c]
            dqdv_r = cg['discharge_dQdV']
            for t in range(min(T, dqdv_r.shape[0])):
                dqdv = f_orig[dqdv_r[t, 0]][()].flatten()
                if len(dqdv) < 10:
                    continue
                FEAT_D[idx, t, 0] = np.var(dqdv)
                peaks, _ = find_peaks(-dqdv, prominence=0.01)
                FEAT_D[idx, t, 1] = len(peaks)
                if len(peaks) > 0:
                    ph = dqdv[peaks]
                    mi = int(np.argmax(np.abs(ph)))
                    FEAT_D[idx, t, 2] = ph[mi]
                    FEAT_D[idx, t, 3] = peaks[mi] / 1000.0
                FEAT_D[idx, t, 4] = skew(dqdv)

    close_data(f_orig, f_proc)

    # 拼接 Feature B/C/D
    FEAT_B = np.concatenate([FEAT_A,
                             build_static_seq(C1s, T),
                             build_static_seq(Q1s, T),
                             build_static_seq(C2s, T)], axis=2)
    FEAT_C = np.concatenate([FEAT_B] + [build_static_seq(crate_bins[:, j], T) for j in range(4)], axis=2)
    FEAT_D_all = np.concatenate([FEAT_C, FEAT_D], axis=2)

    FEATURES = {'A': FEAT_A, 'B': FEAT_B, 'C': FEAT_C, 'D': FEAT_D_all}
    return FEATURES, y_life, meta


# ============================================================
# 时序 → 标量聚合（Ridge/RF/SHAP 用）
# ============================================================
def aggregate_features(X, dyn_names=DYNAMIC_NAMES, static_first_dims=(6, 7, 8),
                       soc_dims=(9, 10, 11, 12), dqdv_dims=(13, 14, 15, 16, 17)):
    """
    把 (N,T,D) 时序压缩为二维标量特征矩阵，返回 (X_agg, feature_names)。

    默认按 Feature D 的维度布局（0-5 动态、6-8 静态、9-12 SOC、13-17 dQ/dV）：
      - 动态特征 × {均值, 标准差, 最小值, 最大值, 斜率} = 6×5 = 30
      - 静态策略参数（取首步值）3
      - SOC bin C-rate（取首步值）4
      - dQ/dV 统计量跨循环均值 5
    """
    N, T, D = X.shape
    feats = []
    names = []

    for d, nm in enumerate(dyn_names):
        v = X[:, :, d]
        feats.extend([np.mean(v, 1), np.std(v, 1), np.min(v, 1), np.max(v, 1),
                      np.array([np.polyfit(np.arange(T), v[i], 1)[0] for i in range(N)])])
        for s in ['mean', 'std', 'min', 'max', 'slope']:
            names.append(f'{nm}_{s}')

    for d in static_first_dims:
        feats.append(X[:, 0, d])
    names.extend(['C1', 'Q1', 'C2'])

    for d in soc_dims:
        feats.append(X[:, 0, d])
    for lo, hi in SOC_BINS:
        names.append(f'Crate_{lo}-{hi}')

    for d in dqdv_dims:
        feats.append(np.mean(X[:, :, d], 1))
    for s in ['方差', '峰数', '峰高', '峰位', '偏度']:
        names.append(f'dQdV_{s}')

    return np.column_stack(feats), names


def flatten_sequence(X):
    """把 (N,T,D) 展平为 (N,T*D)，作为 Ridge/RF 的公平输入（与 LSTM/Transformer 同信息量）。"""
    return X.reshape(X.shape[0], -1)


# ============================================================
# PyTorch 模型
# ============================================================
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


# 延迟构建模型类：只有真正需要 torch 的脚本才 import torch
def build_models():
    torch, nn = _torch()

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=200):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
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
            self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                                    nn.Dropout(0.1), nn.Linear(32, 1))

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            out = self.transformer(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                                dropout=dropout)
            self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                    nn.Dropout(0.1), nn.Linear(32, 1))

        def forward(self, x):
            out, (h, c) = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return PositionalEncoding, TransformerPredictor, LSTMPredictor


def extract_attention_weights(model, x):
    """
    对已训练的 TransformerPredictor，手动重放 encoder 各层以收集注意力权重。

    返回: list，每层一个 (B, nhead, T, T) 张量（第 0 层在前）。
    复现 PyTorch TransformerEncoderLayer 默认 post-LN 前向（norm_first=False），
    仅把 self_attn 改为 need_weights=True 以截获每头权重。
    """
    torch, nn = _torch()
    model.eval()
    x = model.input_proj(x)
    x = model.pos_enc(x)
    attn_list = []
    for layer in model.transformer.layers:
        out, attn = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = layer.norm1(x + layer.dropout1(out))
        x = layer.norm2(x + layer.dropout2(
            layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
        attn_list.append(attn.detach())
    return attn_list


def attention_rollout(model, x):
    """
    Abnar & Zuidema (2020) 的注意力 rollout：逐层做「残差 + 头平均」后连乘，
    得到输入各时间步对最终预测的相对贡献矩阵 (T, T)。
    """
    torch, nn = _torch()
    attn_list = extract_attention_weights(model, x)   # 每层 (B, nhead, T, T)
    B = attn_list[0].shape[0]
    T = attn_list[0].shape[2]
    eye = torch.eye(T).unsqueeze(0).expand(B, -1, -1).to(attn_list[0].device)
    rollout = eye
    for attn in attn_list:
        a = attn.mean(dim=1)            # 头平均 → (B, T, T)
        a = 0.5 * (a + eye)             # 残差连接
        rollout = torch.matmul(a, rollout)
    return rollout[0].cpu().numpy()     # (T, T)


def standardize_sequences(X_tr, X_val):
    """逐特征 Z-score 标准化（统计量来自训练集），返回 (X_tr_s, X_val_s)。"""
    x_mean = X_tr.mean(axis=(0, 1), keepdims=True)
    x_std = X_tr.std(axis=(0, 1), keepdims=True)
    x_std[x_std == 0] = 1.0
    return (X_tr - x_mean) / x_std, (X_val - x_mean) / x_std


def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu',
                return_best=True):
    """
    通用 PyTorch 训练循环（LSTM/Transformer 共用），带早停回滚到最优验证损失。
    return_best=True 时返回验证集损失最低的权重（修正原先「返回最后 epoch、无早停」的缺陷）。
    """
    torch, nn = _torch()
    import copy
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
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
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        scheduler.step(val_loss)

        if return_best and val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if return_best and best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 统计显著性（FDR 校正 + Fisher r-to-z 检验）
# ============================================================
def bh_fdr(pvalues):
    """Benjamini-Hochberg FDR 校正，返回与输入等长的校正后 p 值。"""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    # 秩校正（在升序上计算）
    adj_sorted = sorted_p * n / (np.arange(n) + 1.0)
    # 从后往前做单调非减修正（标准 BH 步骤）
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    # 散射回原始顺序
    adj = np.empty(n)
    adj[order] = adj_sorted
    return np.minimum(adj, 1.0)


def fisher_r_to_z(r1, n1, r2, n2):
    """
    Fisher r-to-z 检验：两个独立样本相关系数是否显著不同。

    返回 (z, p)。用于检验「交互效应」：按某变量分两组后，另一变量与寿命的
    Spearman ρ 在两组间是否存在显著差异（替代原先无检验的 |Δρ|）。
    """
    from scipy.stats import norm
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


# ============================================================
# metrics.json 读写（图/表/报告的数字单一来源）
# ============================================================
def save_metrics(section, data):
    """把某脚本的结果写入 results/metrics.json 的对应 section（合并式更新）。"""
    os.makedirs(results_dir(), exist_ok=True)
    path = os.path.join(results_dir(), 'metrics.json')
    all_m = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fp:
            all_m = json.load(fp)
    all_m[section] = data
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(all_m, fp, indent=2, ensure_ascii=False)
    print(f'  已写入 results/metrics.json [{section}]')


def load_metrics(section=None):
    path = os.path.join(results_dir(), 'metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fp:
        all_m = json.load(fp)
    return all_m if section is None else all_m.get(section, {})


def _to_serializable(obj):
    """把 numpy 标量/数组转成 JSON 可序列化的 Python 类型。"""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
