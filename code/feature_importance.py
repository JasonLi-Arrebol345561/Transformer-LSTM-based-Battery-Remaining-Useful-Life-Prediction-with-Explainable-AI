# -*- coding: utf-8 -*-
"""
=============================================================================
各因素主效应与交互作用对循环寿命的影响（带显著性检验）
=============================================================================
  主效应：Spearman 秩相关 ρ，附 FDR 校正 p。
  交互效应：按调节变量中位数分两组，比较目标变量与寿命的 Spearman ρ，
            用 Fisher r-to-z 检验两组 ρ 是否显著不同（修正原先无检验的 |Δρ|）。
  全部检验统一做 Benjamini-Hochberg FDR 校正。

  说明：主效应 |ρ| 与交互 |Δρ| 是不同含义的统计量，本脚本不再把二者相加进
        同一张排序表，而是分列呈现（各自附显著性），避免「启发式合成」。
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
import warnings
warnings.filterwarnings('ignore')

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

C1s, Q1s, C2s, lifes, tcts = [], [], [], [], []
for g in bats:
    bg = bi[g]
    c1 = float(bg['C1'][()]); q1 = float(bg['Q1'][()]); c2 = float(bg['C2'][()])
    cl = int(bg['cycle_life'][()])
    tct = float(bg['total_charge_time'][()])
    if c1 > 0 and q1 > 0 and c2 > 0:
        C1s.append(c1); Q1s.append(q1); C2s.append(c2); lifes.append(cl); tcts.append(tct)
f.close()

C1 = np.array(C1s); Q1 = np.array(Q1s); C2 = np.array(C2s)
life = np.array(lifes, dtype=float)
avg_ct = np.array(tcts) / life           # 单次平均充电时间
C1_C2_gap = np.abs(C1 - C2)              # 倍率跳变幅度
n = len(life)
print(f'电池数: {n}')

# ============================================================
# 主效应
# ============================================================
main_effects = {
    'Q1 (切换SOC)': (Q1, life),
    'C2 (第二步倍率)': (C2, life),
    'C1 (第一步倍率)': (C1, life),
    '平均充电时间': (avg_ct, life),
    '|C1-C2| 倍率跳变': (C1_C2_gap, life),
}
main_results = {}   # name -> (rho, p)
for name, (x, y) in main_effects.items():
    rho, p = spearmanr(x, y)
    main_results[name] = (rho, p)

# ============================================================
# 交互效应（分层 Spearman + Fisher z 检验）
# ============================================================
def interaction(moderator, x, moderator_name, x_name):
    med = np.median(moderator)
    hi = moderator >= med
    lo = moderator < med
    rho_hi, _ = spearmanr(x[hi], life[hi])
    rho_lo, _ = spearmanr(x[lo], life[lo])
    z, p = C.fisher_r_to_z(rho_hi, int(hi.sum()), rho_lo, int(lo.sum()))
    return {'moderator': moderator_name, 'x': x_name,
            'rho_high': float(rho_hi), 'rho_low': float(rho_lo),
            'delta_rho': float(abs(rho_hi - rho_lo)),
            'z': z, 'p': p}

interactions = {
    'C1 × Q1': interaction(C1, Q1, 'C1', 'Q1'),
    'Q1 × C2': interaction(Q1, C2, 'Q1', 'C2'),
    'C1-C2跳变 × Q1': interaction(C1_C2_gap, Q1, '|C1-C2|', 'Q1'),
    'Q1 × 充电时间': interaction(avg_ct, Q1, '充电时间', 'Q1'),
}

# ============================================================
# FDR 校正（所有 5 主效应 + 4 交互，共 9 项）
# ============================================================
main_names = list(main_results.keys())
inter_names = list(interactions.keys())
all_p = np.array([main_results[k][1] for k in main_names] + [interactions[k]['p'] for k in inter_names])
adj_p = C.bh_fdr(all_p)
main_adj = dict(zip(main_names, adj_p[:len(main_names)]))
inter_adj = dict(zip(inter_names, adj_p[len(main_names):]))


def sig_label(p, adj):
    if adj < 0.05:
        return '显著(FDR<0.05)'
    return '不显著' if p >= 0.05 else '名义显著(FDR后不显著)'


# ============================================================
# 输出
# ============================================================
print('\n' + '=' * 60)
print('1. 主效应 Spearman ρ（FDR 校正）')
print('=' * 60)
print(f'  {"因素":<18} {"ρ":>8} {"原始 p":<10} {"FDR p":<10} {"结论":<18}')
for name in main_names:
    rho, p = main_results[name]
    print(f'  {name:<18} {rho:>+8.3f} {p:<10.4g} {main_adj[name]:<10.4g} {sig_label(p, main_adj[name]):<18}')

print('\n' + '=' * 60)
print('2. 交互效应（分层 Spearman + Fisher r-to-z 检验）')
print('=' * 60)
print(f'  {"交互":<16} {"ρ(高)":>8} {"ρ(低)":>8} {"|Δρ|":>8} {"z":>7} {"p":<9} {"FDR p":<10}')
for name in inter_names:
    d = interactions[name]
    print(f'  {name:<16} {d["rho_high"]:>+8.3f} {d["rho_low"]:>+8.3f} {d["delta_rho"]:>8.3f} '
          f'{d["z"]:>7.2f} {d["p"]:<9.4g} {inter_adj[name]:<10.4g}')

# ============================================================
# 图：主效应与交互分列（不同量纲，分开画）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左：主效应 |ρ|
ax = axes[0]
names_main = list(reversed(main_names))
vals_main = [abs(main_results[k][0]) for k in names_main]
colors_main = ['#2E86AB' if main_adj[k] < 0.05 else '#B0B0B0' for k in names_main]
ax.barh(names_main, vals_main, color=colors_main, edgecolor='white')
for i, k in enumerate(names_main):
    rho, p = main_results[k]
    ax.text(vals_main[i] + 0.01, i, f'{rho:+.3f} {"*" if main_adj[k] < 0.05 else ""}',
            va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('主效应 |Spearman ρ|')
ax.set_title('主效应（* = FDR<0.05）')
ax.grid(axis='x', alpha=0.3)

# 右：交互 |Δρ|
ax = axes[1]
names_inter = list(reversed(inter_names))
vals_inter = [interactions[k]['delta_rho'] for k in names_inter]
colors_inter = ['#FF6B6B' if inter_adj[k] < 0.05 else '#E0A0A0' for k in names_inter]
ax.barh(names_inter, vals_inter, color=colors_inter, edgecolor='white')
for i, k in enumerate(names_inter):
    d = interactions[k]
    ax.text(vals_inter[i] + 0.01, i, f'{d["delta_rho"]:.3f} {"*" if inter_adj[k] < 0.05 else ""}',
            va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('交互 |Δρ|（两组子样本 ρ 之差）')
ax.set_title('交互效应（* = Fisher z 检验 FDR<0.05）')
ax.grid(axis='x', alpha=0.3)

fig.suptitle('因素与交互作用对循环寿命的影响（分列呈现，含显著性）', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(C.fig_dir(), 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 落盘
# ============================================================
C.save_metrics('feature_importance', {
    'main_effects': {k: {'rho': float(main_results[k][0]), 'p': float(main_results[k][1]),
                          'fdr_p': float(main_adj[k])} for k in main_names},
    'interactions': {k: dict(interactions[k], fdr_p=float(inter_adj[k])) for k in inter_names},
})
print('\n全部完成!')
