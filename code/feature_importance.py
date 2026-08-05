# -*- coding: utf-8 -*-
"""
Q5: 各因素及交互作用对循环寿命的影响程度排序
方法: Spearman 主效应 + 分层 Spearman 交互分析
     小样本 (n=124) 下比回归模型更稳健
"""

import h5py
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

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

C1s, Q1s, C2s, lifes, tcts = [], [], [], [], []
for g in bats:
    bg = bi[g]
    c1 = float(bg['C1'][()])
    q1 = float(bg['Q1'][()])
    c2 = float(bg['C2'][()])
    cl = int(bg['cycle_life'][()])
    tct = float(bg['total_charge_time'][()])
    if c1 > 0 and q1 > 0 and c2 > 0:
        C1s.append(c1); Q1s.append(q1); C2s.append(c2)
        lifes.append(cl); tcts.append(tct)
f.close()

C1 = np.array(C1s); Q1 = np.array(Q1s); C2 = np.array(C2s)
life = np.array(lifes, dtype=float)
avg_ct = np.array(tcts) / life           # 单次平均充电时间
C1_C2_gap = np.abs(C1 - C2)              # 倍率跳变幅度

n = len(life)
print(f'电池数: {n}')

# ============================================================
# 主效应 Spearman (与之前一致，这里汇总)
# ============================================================
print()
print('=' * 60)
print('1. 主效应 Spearman ρ')
print('=' * 60)

main_effects = {
    'Q1 (切换SOC)':      (Q1, life),
    'C2 (第二步倍率)':    (C2, life),
    'C1 (第一步倍率)':    (C1, life),
    '平均充电时间':       (avg_ct, life),
    '|C1-C2| 倍率跳变':  (C1_C2_gap, life),
}

main_results = []
for name, (x, y) in main_effects.items():
    rho, p = spearmanr(x, y)
    main_results.append((name, abs(rho), rho, p))
    if p < 0.001: sig = 'p < 0.001'
    elif p < 0.01: sig = 'p < 0.01'
    elif p < 0.05: sig = 'p < 0.05'
    else: sig = f'p = {p:.2f}'
    print(f'  {name:20s}  |ρ| = {abs(rho):.4f}  ({sig})')

# ============================================================
# 2. 交互效应: 分层 Spearman
# ============================================================
print()
print('=' * 60)
print('2. 交互效应 (分层 Spearman)')
print('=' * 60)

# --- Q1 × C2 交互: 高 Q1 vs 低 Q1 组内 C2 的效应差异 ---
q1_med = np.median(Q1)
mask_high_q1 = Q1 >= q1_med
mask_low_q1  = Q1 < q1_med

rho_c2_high_q1, p_high = spearmanr(C2[mask_high_q1], life[mask_high_q1])
rho_c2_low_q1,  p_low  = spearmanr(C2[mask_low_q1],  life[mask_low_q1])
interaction_q1_c2 = abs(rho_c2_high_q1 - rho_c2_low_q1)

print(f'  Q1 × C2 交互:')
print(f'    高 Q1 (≥{q1_med:.0f}%) 组内 C2 效应: ρ = {rho_c2_high_q1:+.4f}')
print(f'    低 Q1 (<{q1_med:.0f}%) 组内 C2 效应: ρ = {rho_c2_low_q1:+.4f}')
print(f'    交互强度 (|Δρ|) = {interaction_q1_c2:.4f}')

# --- C1 × Q1 交互: 高 C1 vs 低 C1 组内 Q1 的效应差异 ---
c1_med = np.median(C1)
mask_high_c1 = C1 >= c1_med
mask_low_c1  = C1 < c1_med

rho_q1_high_c1, p_h = spearmanr(Q1[mask_high_c1], life[mask_high_c1])
rho_q1_low_c1,  p_l = spearmanr(Q1[mask_low_c1],  life[mask_low_c1])
interaction_c1_q1 = abs(rho_q1_high_c1 - rho_q1_low_c1)

print()
print(f'  C1 × Q1 交互:')
print(f'    高 C1 (≥{c1_med:.1f}C) 组内 Q1 效应: ρ = {rho_q1_high_c1:+.4f}')
print(f'    低 C1 (<{c1_med:.1f}C) 组内 Q1 效应: ρ = {rho_q1_low_c1:+.4f}')
print(f'    交互强度 (|Δρ|) = {interaction_c1_q1:.4f}')

# --- C1 × C2 交互: 倍率一致性 ---
# 高 C1_C2_gap vs 低 C1_C2_gap 组内 Q1 的效应差异
gap_med = np.median(C1_C2_gap)
mask_high_gap = C1_C2_gap >= gap_med
mask_low_gap  = C1_C2_gap < gap_med

rho_q1_high_gap, p_hg = spearmanr(Q1[mask_high_gap], life[mask_high_gap])
rho_q1_low_gap,  p_lg = spearmanr(Q1[mask_low_gap],  life[mask_low_gap])
interaction_gap = abs(rho_q1_high_gap - rho_q1_low_gap)

print()
print(f'  C1-C2 一致性 × Q1 交互:')
print(f'    倍率跳变大 (≥{gap_med:.1f}C) 组内 Q1 效应: ρ = {rho_q1_high_gap:+.4f}')
print(f'    倍率跳变小 (<{gap_med:.1f}C) 组内 Q1 效应: ρ = {rho_q1_low_gap:+.4f}')
print(f'    交互强度 (|Δρ|) = {interaction_gap:.4f}')

# --- Q1 × 充电时间 交互 ---
ct_med = np.median(avg_ct)
mask_high_ct = avg_ct >= ct_med
mask_low_ct  = avg_ct < ct_med

rho_q1_high_ct, p_hc = spearmanr(Q1[mask_high_ct], life[mask_high_ct])
rho_q1_low_ct,  p_lc = spearmanr(Q1[mask_low_ct],  life[mask_low_ct])
interaction_q1_ct = abs(rho_q1_high_ct - rho_q1_low_ct)

print()
print(f'  Q1 × 充电时间 交互:')
print(f'    长充电 (≥{ct_med:.1f}h/次) 组内 Q1 效应: ρ = {rho_q1_high_ct:+.4f}')
print(f'    短充电 (<{ct_med:.1f}h/次) 组内 Q1 效应: ρ = {rho_q1_low_ct:+.4f}')
print(f'    交互强度 (|Δρ|) = {interaction_q1_ct:.4f}')

# ============================================================
# 3. 综合排名 (主效应 + 交互)
# ============================================================
print()
print('=' * 60)
print('3. 综合排名: 所有因素对寿命的影响程度')
print('=' * 60)

# 主效应权重: 用 |ρ|
# 交互效应权重: 用 |Δρ| (分层 Spearman 差异绝对值)
# 总贡献 = 主效应 + 交互效应 (两者量纲可比，直接加)

all_factors = []

# 主效应
for name, abs_rho, rho, p in main_results:
    all_factors.append((name, '主效应', abs_rho, 0.0, abs_rho))

# 交互效应 (按强度)
interactions = [
    ('Q1 × C2: 切换SOC调节第二步倍率效应', interaction_q1_c2),
    ('C1 × Q1: 第一步倍率调节切换SOC效应',  interaction_c1_q1),
    ('C1-C2跳变 × Q1: 倍率一致性调节切换SOC效应', interaction_gap),
    ('Q1 × 充电时间: 充电速度调节切换SOC效应', interaction_q1_ct),
]

for name, strength in interactions:
    all_factors.append((name, '交互', 0.0, strength, strength))

# 按总贡献排序
all_factors.sort(key=lambda x: x[4], reverse=True)

print()
print(f'  {"排名":<4} {"因素":<45} {"类型":<6} {"主效应 |ρ|":<12} {"交互 |Δρ|":<12} {"总贡献"}')
print(f'  {"-"*4} {"-"*45} {"-"*6} {"-"*12} {"-"*12} {"-"*8}')

for rank, (name, ftype, main, inter, total) in enumerate(all_factors, 1):
    print(f'  {rank:<4} {name:<45} {ftype:<6} {main:<12.4f} {inter:<12.4f} {total:<8.4f}')

# ============================================================
# 图表: 综合排名柱状图
# ============================================================
# 把主效应和交互合并展示
plot_items = []
for name, ftype, main, inter, total in all_factors:
    if ftype == '主效应':
        plot_items.append((name, main, '#2E86AB'))
    else:
        plot_items.append((name, inter, '#FF6B6B'))

# 按贡献排序
plot_items.sort(key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(11, 5.5))
names = [p[0] for p in plot_items]
values = [p[1] for p in plot_items]
colors = [p[2] for p in plot_items]

bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('影响强度 (Spearman |ρ| 或 交互 |Δρ|)', fontsize=12)
ax.set_title('各因素及交互作用对循环寿命的影响程度排序\n'
             '(主效应 = Spearman |ρ|,  交互 = 分层 Spearman |Δρ|)',
             fontsize=12, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='主效应 (Spearman |ρ|)'),
    Patch(facecolor='#FF6B6B', label='交互效应 (分层 |Δρ|)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n图表保存: {FIG_DIR}/feature_importance.png')
print('\n全部完成!')
