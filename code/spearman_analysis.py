# -*- coding: utf-8 -*-
"""
=============================================================================
Spearman 相关性分析：充电策略参数 vs 循环寿命
=============================================================================
  研究问题：
    Q1-Q3: C1 / Q1 / C2 对寿命的影响
    Q4   : 不同 SOC 区间高倍率充电对寿命的影响差异

  方法：Spearman 秩相关；SOC 区间等效 C-rate 按「SOC 覆盖宽度」加权平均
        （修正原先的简单均值）；对所有检验做 Benjamini-Hochberg FDR 校正。
  输出：figures/spearman_bar.png、figures/spearman_scatter.png，及 metrics.json。
=============================================================================
"""

import h5py
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import common as C

os.makedirs(C.fig_dir(), exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 读取数据
# ============================================================
f = h5py.File(C.mat_proc_path(), 'r')
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
        C1_vals.append(c1); Q1_vals.append(q1); C2_vals.append(c2); lifespans.append(cl)
f.close()

C1 = np.array(C1_vals); Q1 = np.array(Q1_vals); C2 = np.array(C2_vals)
life = np.array(lifespans, dtype=float)
print(f'读取 {len(life)} 个电池（策略参数有效）')


def report(label, x, y):
    rho, p = spearmanr(x, y)
    sig = 'p<0.001' if p < 0.001 else ('p<0.01' if p < 0.01 else ('p<0.05' if p < 0.05 else f'p={p:.2f}'))
    print(f'  {label}: ρ = {rho:+.4f}, {sig}')
    return rho, p


# ============================================================
# 主效应 Spearman
# ============================================================
print('\n--- 主效应：单一参数 vs 循环寿命 ---')
rho_c1, p_c1 = report('C1', C1, life)
rho_q1, p_q1 = report('Q1', Q1, life)
rho_c2, p_c2 = report('C2', C2, life)

# ============================================================
# SOC 区间 C-rate（宽度加权，5 bins）
# ============================================================
print('\n--- SOC 区间 C-rate vs 循环寿命（宽度加权） ---')
SOC_BINS = C.SOC_BINS_ALL
bin_crates = np.zeros((len(life), len(SOC_BINS)))
for i in range(len(life)):
    bin_crates[i] = C.soc_bin_crates(C1[i], Q1[i], C2[i], SOC_BINS)

bin_rhos, bin_ps = [], []
for j, (lo, hi) in enumerate(SOC_BINS):
    x = bin_crates[:, j]
    valid = ~np.isnan(x)
    if np.std(x[valid]) < 1e-10:
        bin_rhos.append(np.nan); bin_ps.append(np.nan)
        print(f'  SOC [{lo}-{hi}%]: ρ = N/A (常数 C-rate，Spearman 无定义)')
    else:
        rho, p = spearmanr(x[valid], life[valid])
        bin_rhos.append(rho); bin_ps.append(p)
        report(f'SOC [{lo}-{hi}%]', x[valid], life[valid])

# ============================================================
# FDR 校正（所有 8 项检验一起校正）
# ============================================================
print('\n--- Benjamini-Hochberg FDR 校正 ---')
all_labels = ['C1', 'Q1', 'C2'] + [f'SOC {lo}-{hi}%' for lo, hi in SOC_BINS]
all_p = np.array([p_c1, p_q1, p_c2] + bin_ps)
valid_mask = ~np.isnan(all_p)
adj_p = np.full(len(all_p), np.nan)
adj_p[valid_mask] = C.bh_fdr(all_p[valid_mask])

print(f'  {"检验":<16} {"原始 p":<12} {"FDR 校正 p":<12} {"显著(FDR<0.05)":<12}')
for i, lbl in enumerate(all_labels):
    if np.isnan(all_p[i]):
        continue
    sig = '是' if adj_p[i] < 0.05 else '否'
    print(f'  {lbl:<16} {all_p[i]:<12.4g} {adj_p[i]:<12.4g} {sig:<12}')

# ============================================================
# 图1: SOC 区间 ρ 柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
valid_bins = [(i, l, r) for i, (l, r) in enumerate(zip(all_labels[3:], bin_rhos)) if not np.isnan(r)]
colors = ['#2E86AB' if r < 0 else '#A23B72' for _, _, r in valid_bins]
bars = ax.bar([l for _, l, _ in valid_bins], [r for _, _, r in valid_bins],
              color=colors, edgecolor='white', linewidth=1.2)
for bar, (_, _, rho), (i, _, _) in zip(bars, valid_bins, [(i, l, r) for i, (l, r) in enumerate(zip(all_labels[3:], bin_rhos)) if not np.isnan(r)]):
    p = bin_ps[i]
    sig = 'p<0.001' if p < 0.001 else ('p<0.01' if p < 0.01 else ('p<0.05' if p < 0.05 else ''))
    y_pos = bar.get_height()
    off = 0.02 if y_pos >= 0 else -0.08
    label = f'ρ={rho:.3f} ({sig})' if sig else f'ρ={rho:.3f}'
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos + off, label, ha='center',
            va='bottom' if y_pos >= 0 else 'top', fontsize=10, fontweight='bold')
fig.text(0.5, 0.01, '* SOC 80–100%: 所有电池均为 1C CC-CV，C-rate 恒定，Spearman 无定义',
         ha='center', fontsize=8.5, color='gray', style='italic')
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
ax.set_ylabel('Spearman ρ', fontsize=13)
ax.set_xlabel('SOC 区间', fontsize=13)
ax.set_title('不同 SOC 区间 C-rate 暴露量与循环寿命的 Spearman 相关\n(负值越强 = 该区间高倍率充电对寿命损害越大)',
             fontsize=13, fontweight='bold')
ax.set_ylim(min(r for _, _, r in valid_bins) - 0.15, max(r for _, _, r in valid_bins) + 0.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'spearman_bar.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 图2: 散点 + 滑动窗口平滑（非严格 LOWESS）
# ============================================================
def sliding_mean(x, y, frac=0.5):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return x, y
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    win = max(int(len(x) * frac), 10)
    ysmooth = np.array([np.mean(ys[max(0, i - win // 2):min(len(x), i + win // 2)]) for i in range(len(x))])
    return xs, ysmooth


fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
params = [(C1, 'C1', '第一步充电倍率 (C-rate)', rho_c1, p_c1),
          (Q1, 'Q1', '阶段切换 SOC (%)', rho_q1, p_q1),
          (C2, 'C2', '第二步充电倍率 (C-rate)', rho_c2, p_c2)]
for ax, (x, name, xlabel, rho, p) in zip(axes, params):
    ax.scatter(x, life, c='#4ECDC4', edgecolors='#2C7873', alpha=0.7, s=50, zorder=3)
    xs, ys = sliding_mean(x, life, frac=0.4)
    ax.plot(xs, ys, color='#FF6B6B', linewidth=2.5, zorder=4, label='滑动窗口平滑')
    sig = 'p<0.001' if p < 0.001 else ('p<0.01' if p < 0.01 else ('p<0.05' if p < 0.05 else f'p={p:.2f}'))
    ax.text(0.05, 0.92, f'Spearman ρ = {rho:+.3f} ({sig})', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel('循环寿命', fontsize=12)
    ax.set_title(f'{name} vs 循环寿命', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'spearman_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 落盘
# ============================================================
C.save_metrics('spearman', {
    'main_effects': {
        'C1': {'rho': float(rho_c1), 'p': float(p_c1)},
        'Q1': {'rho': float(rho_q1), 'p': float(p_q1)},
        'C2': {'rho': float(rho_c2), 'p': float(p_c2)},
    },
    'soc_bins': {
        f'{lo}-{hi}%': {'rho': float(rho), 'p': float(p)}
        for (lo, hi), rho, p in zip(SOC_BINS, bin_rhos, bin_ps)
    },
    'fdr_adjusted_p': {lbl: (float(adj_p[i]) if not np.isnan(adj_p[i]) else None)
                        for i, lbl in enumerate(all_labels)},
})
print('\n全部完成!')
