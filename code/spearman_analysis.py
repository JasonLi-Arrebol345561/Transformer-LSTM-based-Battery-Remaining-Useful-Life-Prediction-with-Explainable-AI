# -*- coding: utf-8 -*-
"""
Spearman 相关性分析：充电策略参数 vs 循环寿命
Q1: C1 对寿命的影响
Q2: Q1 对寿命的影响
Q3: C2 对寿命的影响
Q4: 不同 SOC 区间高倍率充电对寿命的影响差异
"""

import h5py
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# 配置
# ============================================================
MAT_FILE = r'E:\VSCode\BatteryLife-AI\battery_analysis_results.mat'
FIG_DIR = r'E:\VSCode\BatteryLife-AI\figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 读取数据
# ============================================================
f = h5py.File(MAT_FILE, 'r')
bi = f['battery_info']
bats = sorted(bi.keys())

C1_vals, Q1_vals, C2_vals, lifespans = [], [], [], []

for g in bats:
    bg = bi[g]
    c1 = float(bg['C1'][()])
    q1 = float(bg['Q1'][()])
    c2 = float(bg['C2'][()])
    cl = int(bg['cycle_life'][()])

    if c1 > 0:
        C1_vals.append(c1)
        Q1_vals.append(q1)
        C2_vals.append(c2)
        lifespans.append(cl)

f.close()

C1 = np.array(C1_vals)
Q1 = np.array(Q1_vals)
C2 = np.array(C2_vals)
life = np.array(lifespans, dtype=float)

print(f'读取 {len(life)} 个电池')

# ============================================================
# Q1-Q3: Spearman 相关性
# ============================================================
print()
print('=' * 60)
print('Spearman 秩相关分析')
print('=' * 60)

def report(label, x, y):
    rho, p = spearmanr(x, y)
    if p < 0.001:
        sig = 'p < 0.001'
    elif p < 0.01:
        sig = 'p < 0.01'
    elif p < 0.05:
        sig = 'p < 0.05'
    else:
        sig = '不显著 (p >= 0.05)'
    direction = '正' if rho > 0 else '负'
    print(f'  {label}:')
    print(f'    ρ = {rho:+.4f},  {sig}')
    print(f'    解读: {direction}相关, ', end='')
    if p < 0.05:
        if abs(rho) > 0.5:
            print('强显著')
        elif abs(rho) > 0.3:
            print('中等显著')
        else:
            print('弱显著')
    else:
        print('不显著')
    return rho, p

print()
print('--- Q1-Q3: 单一参数 vs 循环寿命 ---')
rho_c1, p_c1 = report('C1 (第一步倍率) vs 循环寿命', C1, life)
rho_q1, p_q1 = report('Q1 (切换SOC%) vs 循环寿命', Q1, life)
rho_c2, p_c2 = report('C2 (第二步倍率) vs 循环寿命', C2, life)

# ============================================================
# Q4: 不同 SOC 区间的 C-rate 暴露量 vs 循环寿命
# ============================================================
print()
print('--- Q4: 各 SOC 区间的 C-rate vs 循环寿命 ---')

# 5 个 SOC bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
SOC_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
N_BIN = len(SOC_BINS)

# 对每个电池，计算每个 SOC bin 中的等效 C-rate
# 根据策略解析 C-rate 在各 bin 中的分布
bin_crates = np.zeros((len(life), N_BIN))

for i in range(len(life)):
    c1 = C1[i]
    q1 = Q1[i]
    c2 = C2[i]

    for j, (lo, hi) in enumerate(SOC_BINS):
        # 该 bin 落在哪个充电阶段？
        # 阶段1: [0, q1] at C1
        # 阶段2: [q1, 80] at C2
        # 阶段3: [80, 100] at 1C (CC-CV)

        # 计算该 bin 中各阶段覆盖的比例
        crates_in_bin = []

        # 阶段1 覆盖部分
        s1_lo = max(lo, 0)
        s1_hi = min(hi, q1)
        if s1_hi > s1_lo:
            crates_in_bin.append(c1)

        # 阶段2 覆盖部分
        s2_lo = max(lo, q1)
        s2_hi = min(hi, 80)
        if s2_hi > s2_lo:
            crates_in_bin.append(c2)

        # 阶段3 覆盖部分
        s3_lo = max(lo, 80)
        s3_hi = min(hi, 100)
        if s3_hi > s3_lo:
            crates_in_bin.append(1.0)  # CC-CV 近似 1C

        if crates_in_bin:
            # 按覆盖 SOC 宽度加权平均
            bin_crates[i, j] = np.mean(crates_in_bin)
        else:
            bin_crates[i, j] = np.nan

bin_rhos = []
bin_ps = []
bin_labels = [f'{lo}–{hi}%' for lo, hi in SOC_BINS]

for j, label in enumerate(bin_labels):
    valid = ~np.isnan(bin_crates[:, j])
    x = bin_crates[valid, j]
    # 检测常数输入（如 SOC 80-100% 全是 1C）
    if np.std(x) < 1e-10:
        rho, p = np.nan, np.nan
        bin_rhos.append(np.nan)
        bin_ps.append(np.nan)
        print(f'  SOC [{label}] C-rate vs 循环寿命:')
        print(f'    ρ = N/A (常数 C-rate = {x[0]:.1f}C，Spearman 无定义)')
    else:
        rho, p = spearmanr(x, life[valid])
        bin_rhos.append(rho)
        bin_ps.append(p)
        report(f'SOC [{label}] C-rate vs 循环寿命', x, life[valid])

# ============================================================
# 图1: 柱状图 — 各 SOC 区间的 Spearman ρ
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

# 只画有效的 SOC bins (排除 NaN)
valid_bins = [(i, l, r) for i, (l, r) in enumerate(zip(bin_labels, bin_rhos)) if not np.isnan(r)]
valid_labels = [v[1] for v in valid_bins]
valid_rhos = [v[2] for v in valid_bins]
valid_ps = [bin_ps[v[0]] for v in valid_bins]

colors = ['#2E86AB' if r < 0 else '#A23B72' for r in valid_rhos]
bars = ax.bar(valid_labels, valid_rhos, color=colors, edgecolor='white', linewidth=1.2)

# 标注 ρ 值
for bar, rho, p in zip(bars, valid_rhos, valid_ps):
    if p < 0.001:
        sig = 'p<0.001'
    elif p < 0.01:
        sig = 'p<0.01'
    elif p < 0.05:
        sig = 'p<0.05'
    else:
        sig = ''
    y_pos = bar.get_height()
    offset = 0.02 if y_pos >= 0 else -0.08
    label = f'ρ={rho:.3f} ({sig})' if sig else f'ρ={rho:.3f}'
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
            label, ha='center', va='bottom' if y_pos >= 0 else 'top',
            fontsize=10, fontweight='bold')

# 添加 SOC 80-100% 的说明（放在图下方，不占用 axes 空间）
fig.text(0.5, 0.01, '* SOC 80–100%: 所有电池均为 1C CC-CV，C-rate 恒定，Spearman 无定义',
         ha='center', fontsize=8.5, color='gray', style='italic')

ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
ax.set_ylabel("Spearman ρ", fontsize=13)
ax.set_xlabel('SOC 区间', fontsize=13)
ax.set_title('不同 SOC 区间 C-rate 暴露量与循环寿命的 Spearman 相关\n'
             '(负值越强 = 该区间高倍率充电对寿命损害越大)',
             fontsize=13, fontweight='bold')
rho_vals = [r for r in valid_rhos]
ax.set_ylim(min(rho_vals) - 0.15, max(rho_vals) + 0.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'spearman_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n柱状图保存: {FIG_DIR}/spearman_bar.png')

# ============================================================
# 图2: 散点图 + LOWESS — C1, Q1, C2 vs cycle_life
# ============================================================
from scipy.interpolate import make_interp_spline

def lowess_smooth(x, y, frac=0.5):
    """简易 LOWESS 平滑"""
    # 过滤非有限值
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return x, y
    x_sorted_idx = np.argsort(x)
    x_sorted = x[x_sorted_idx]
    y_sorted = y[x_sorted_idx]

    # 滑动窗口加权平均
    window = max(int(n * frac), 10)
    y_smooth = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2)
        y_smooth[i] = np.mean(y_sorted[lo:hi])

    return x_sorted, y_smooth

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

params = [
    (C1, 'C1', '第一步充电倍率 (C-rate)', rho_c1, p_c1),
    (Q1, 'Q1', '阶段切换 SOC (%)', rho_q1, p_q1),
    (C2, 'C2', '第二步充电倍率 (C-rate)', rho_c2, p_c2),
]

for ax, (x, name, xlabel, rho, p) in zip(axes, params):
    # 散点
    ax.scatter(x, life, c='#4ECDC4', edgecolors='#2C7873', alpha=0.7, s=50, zorder=3)

    # LOWESS 平滑线
    xs, ys = lowess_smooth(x, life, frac=0.4)
    ax.plot(xs, ys, color='#FF6B6B', linewidth=2.5, zorder=4, label='LOWESS 趋势')

    # Spearman 标注
    if p < 0.001:
        sig = 'p<0.001'
    elif p < 0.01:
        sig = 'p<0.01'
    elif p < 0.05:
        sig = 'p<0.05'
    else:
        sig = f'p={p:.2f}'
    ax.text(0.05, 0.92, f'Spearman ρ = {rho:+.3f} ({sig})',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('循环寿命', fontsize=12)
    ax.set_title(f'{name} vs 循环寿命', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'spearman_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'散点图保存: {FIG_DIR}/spearman_scatter.png')

# ============================================================
# 汇总
# ============================================================
print()
print('=' * 60)
print('汇总')
print('=' * 60)

# 按 |ρ| 排序所有分析项
all_results = [
    ('C1', rho_c1, p_c1),
    ('Q1', rho_q1, p_q1),
    ('C2', rho_c2, p_c2),
] + [(f'SOC {l}', r, p) for l, r, p in zip(bin_labels, bin_rhos, bin_ps)]

all_results.sort(key=lambda x: abs(x[1]), reverse=True)

print()
print('  按影响强度排序 (|ρ| 降序):')
for rank, (label, rho, p) in enumerate(all_results, 1):
    if p < 0.001:
        sig = 'p < 0.001'
    elif p < 0.01:
        sig = 'p < 0.01'
    elif p < 0.05:
        sig = 'p < 0.05'
    elif np.isnan(p):
        sig = 'N/A (常数)'
    else:
        sig = f'p = {p:.2f} (不显著)'
    print(f'  {rank}. {label:20s}  ρ = {rho:+.4f}  {sig}')

print()
print('全部完成!')
