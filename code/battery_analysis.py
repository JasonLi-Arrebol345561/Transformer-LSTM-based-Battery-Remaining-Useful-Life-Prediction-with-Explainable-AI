# -*- coding: utf-8 -*-
"""
MIT-Stanford Battery Dataset — 124电池基础信息整理
输出: battery_analysis_results.mat + 箱线图
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import os

# ============================================================
# 配置
# ============================================================
MAT_FILE = r'E:\VSCode\BatteryLife-AI\MIT-Stanford Battery Dataset_cleaned.mat'
OUT_DIR = r'E:\VSCode\BatteryLife-AI'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
OUT_MAT = os.path.join(OUT_DIR, 'battery_analysis_results.mat')

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def decode_policy(f, ref):
    obj = f[ref]
    chars = [chr(int(x[0])) for x in obj[:]]
    return ''.join(chars)


def parse_policy(policy_str):
    s = policy_str.replace('-newstructure', '')
    m = re.match(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C', s)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


# ============================================================
# 步骤 1: 读取数据
# ============================================================
print('=' * 60)
print('步骤 1: 读取数据...')

f = h5py.File(MAT_FILE, 'r')
bc = f['batch_combined']

N_BAT = 124
batteries = []

for i in range(N_BAT):
    ref_p = bc['policy_readable'][i, 0]
    policy_str = decode_policy(f, ref_p)
    c1, q1, c2 = parse_policy(policy_str)

    ref_s = bc['summary'][i, 0]
    summary = f[ref_s]

    n_rows = summary['QDischarge'].shape[1]
    qd_raw = summary['QDischarge'][0, :].copy()
    qc_raw = summary['QCharge'][0, :].copy()
    ct_raw = summary['chargetime'][0, :].copy()
    ir_raw = summary['IR'][0, :].copy()

    bl_ref = bc['cycle_life'][i, 0]
    bat_label_val = f[bl_ref][0, 0]

    if i <= 4:
        cycle_numbers = np.arange(1, n_rows + 1, dtype=np.float64)
    else:
        cycle_numbers = summary['cycle'][0, :].copy()

    soh = qd_raw / 1.1
    cycle_life = n_rows
    total_ct = np.sum(ct_raw)

    batteries.append({
        'bat_label': bat_label_val,
        'policy': policy_str,
        'C1': c1 if c1 else np.nan,
        'Q1': q1 if q1 else np.nan,
        'C2': c2 if c2 else np.nan,
        'cycle_life': cycle_life,
        'total_charge_time': total_ct,
        'QDischarge': qd_raw,
        'SOH': soh,
        'chargetime': ct_raw,
        'cycle_numbers': cycle_numbers,
        'QCharge': qc_raw,
        'IR': ir_raw,
    })

f.close()
print(f'  读取完毕: {len(batteries)} 个电池')
print(f'  Bat 0 cycle: {batteries[0]["cycle_numbers"][0]:.0f} -> {batteries[0]["cycle_numbers"][-1]:.0f}')
print(f'  Bat 5 cycle: {batteries[5]["cycle_numbers"][0]:.0f} -> {batteries[5]["cycle_numbers"][-1]:.0f}')

# ============================================================
# 步骤 2: 统计指标
# ============================================================
print()
print('=' * 60)
print('步骤 2: 统计指标')

cycle_lifes = np.array([b['cycle_life'] for b in batteries])
total_cts = np.array([b['total_charge_time'] for b in batteries])

print(f'  循环寿命: min={cycle_lifes.min():.0f}, max={cycle_lifes.max():.0f}, '
      f'median={np.median(cycle_lifes):.0f}, mean={cycle_lifes.mean():.1f}')
print(f'  总充电时间: min={total_cts.min():.1f}h, max={total_cts.max():.1f}h, '
      f'median={np.median(total_cts):.1f}h, mean={total_cts.mean():.1f}h')

# ============================================================
# 步骤 3: 典型策略
# ============================================================
print()
print('=' * 60)
print('步骤 3: 典型策略分析')

policy_groups = defaultdict(list)
for b in batteries:
    policy_groups[b['policy']].append(b['cycle_life'])

policy_stats = []
for pol, lifes in policy_groups.items():
    policy_stats.append({
        'policy': pol,
        'count': len(lifes),
        'mean_life': np.mean(lifes),
        'min_life': np.min(lifes),
        'max_life': np.max(lifes),
    })

policy_stats.sort(key=lambda x: x['mean_life'], reverse=True)

def get_params(pol_str):
    c1, q1, c2 = parse_policy(pol_str)
    return c1, q1, c2

print()
print('  === 长寿策略 Top 5 ===')
for rank, ps in enumerate(policy_stats[:5], 1):
    c1, q1, c2 = get_params(ps['policy'])
    print(f'  {rank}. "{ps["policy"]}"')
    print(f'     平均寿命: {ps["mean_life"]:.0f} cycles | '
          f'电池数: {ps["count"]} | '
          f'C1={c1}C, Q1={q1}%, C2={c2}C')

print()
print('  === 短寿策略 Top 5 ===')
for rank, ps in enumerate(policy_stats[-5:], 1):
    c1, q1, c2 = get_params(ps['policy'])
    print(f'  {rank}. "{ps["policy"]}"')
    print(f'     平均寿命: {ps["mean_life"]:.0f} cycles | '
          f'电池数: {ps["count"]} | '
          f'C1={c1}C, Q1={q1}%, C2={c2}C')

# ============================================================
# 步骤 4: 输出 .mat
# ============================================================
print()
print('=' * 60)
print('步骤 4: 输出 .mat 文件...')

f_out = h5py.File(OUT_MAT, 'w')
bi_grp = f_out.create_group('battery_info')

for i, b in enumerate(batteries):
    bg = bi_grp.create_group(f'bat_{i+1:03d}')
    bg.create_dataset('bat_label', data=b['bat_label'])
    bg.create_dataset('policy', data=b['policy'].encode('utf-8'))
    bg.create_dataset('C1', data=b['C1'] if not np.isnan(b['C1']) else -1.0)
    bg.create_dataset('Q1', data=b['Q1'] if not np.isnan(b['Q1']) else -1.0)
    bg.create_dataset('C2', data=b['C2'] if not np.isnan(b['C2']) else -1.0)
    bg.create_dataset('cycle_life', data=b['cycle_life'])
    bg.create_dataset('total_charge_time', data=b['total_charge_time'])
    bg.create_dataset('cycle_numbers', data=b['cycle_numbers'].reshape(1, -1))
    bg.create_dataset('QDischarge', data=b['QDischarge'].reshape(1, -1))
    bg.create_dataset('QCharge', data=b['QCharge'].reshape(1, -1))
    bg.create_dataset('SOH', data=b['SOH'].reshape(1, -1))
    bg.create_dataset('chargetime', data=b['chargetime'].reshape(1, -1))
    bg.create_dataset('IR', data=b['IR'].reshape(1, -1))

ps_grp = f_out.create_group('policy_stats')
for i, ps in enumerate(policy_stats):
    pg = ps_grp.create_group(f'policy_{i+1:03d}')
    pg.create_dataset('policy', data=ps['policy'].encode('utf-8'))
    pg.create_dataset('count', data=ps['count'])
    pg.create_dataset('mean_life', data=ps['mean_life'])
    pg.create_dataset('min_life', data=ps['min_life'])
    pg.create_dataset('max_life', data=ps['max_life'])

f_out.create_dataset('num_batteries', data=124)
f_out.create_dataset('all_cycle_lifes', data=cycle_lifes)
f_out.create_dataset('all_total_charge_times', data=total_cts)
f_out.close()
print(f'  已保存到: {OUT_MAT}')

# ============================================================
# 步骤 5: 箱线图
# ============================================================
print()
print('=' * 60)
print('步骤 5: 绘制箱线图...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
bp1 = ax1.boxplot(cycle_lifes, widths=0.4, patch_artist=True,
                   medianprops={'color': 'black', 'linewidth': 2},
                   flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4})
bp1['boxes'][0].set_facecolor('#4ECDC4')
ax1.set_ylabel('循环寿命')
ax1.set_title(f'循环寿命分布 (n={N_BAT})\n'
              f'最小值={cycle_lifes.min():.0f}, 最大值={cycle_lifes.max():.0f}, '
              f'中位数={np.median(cycle_lifes):.0f}')
ax1.set_xticks([])
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
bp2 = ax2.boxplot(total_cts, widths=0.4, patch_artist=True,
                   medianprops={'color': 'black', 'linewidth': 2},
                   flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4})
bp2['boxes'][0].set_facecolor('#FF6B6B')
ax2.set_ylabel('总充电时间 (小时)')
ax2.set_title(f'总充电时间分布 (n={N_BAT})\n'
              f'最小值={total_cts.min():.1f}h, 最大值={total_cts.max():.1f}h, '
              f'中位数={np.median(total_cts):.1f}h')
ax2.set_xticks([])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'boxplots.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  已保存到: {fig_path}')

# ============================================================
# 验证
# ============================================================
print()
print('=' * 60)
print('验证:')
assert len(batteries) == 124, f'电池数错误: {len(batteries)}'
print(f'  OK 电池数 = {len(batteries)}')
assert batteries[0]['cycle_numbers'][0] == 1 and batteries[0]['cycle_numbers'][-1] == 1851
print(f'  OK Bat 0 cycle = 1->{int(batteries[0]["cycle_numbers"][-1])}')
assert batteries[5]['cycle_numbers'][0] == 1
print(f'  OK Bat 5 cycle = 1->{int(batteries[5]["cycle_numbers"][-1])}')
qd_sample = batteries[0]['QDischarge'][1]
soh_sample = batteries[0]['SOH'][1]
assert abs(soh_sample - qd_sample / 1.1) < 1e-10
print(f'  OK SOH = QDischarge / 1.1 (Qd={qd_sample:.4f}, SOH={soh_sample:.4f})')
total_in_stats = sum(ps['count'] for ps in policy_stats)
assert total_in_stats == 124
print(f'  OK 策略分组覆盖 {total_in_stats} 个电池, 共 {len(policy_stats)} 种策略')

print()
print('=' * 60)
print('全部完成!')
print(f'  .mat: {OUT_MAT}')
print(f'  图片: {fig_path}')
