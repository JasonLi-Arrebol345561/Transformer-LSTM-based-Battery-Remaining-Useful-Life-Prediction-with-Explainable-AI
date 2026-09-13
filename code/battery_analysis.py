# -*- coding: utf-8 -*-
"""
=============================================================================
MIT-Stanford Battery Dataset — 124 电池基础信息整理
=============================================================================
  功能：从原始 .mat 文件中提取 124 个电池的基础信息（充电策略、循环寿命、
        总充电时间、SOH/容量/内阻等时间序列），整理成一份轻量的 .mat 结果文件，
        并绘制箱线图用于数据初探。

  输入：data/MIT-Stanford Battery Dataset_cleaned.mat（原始数据集）
  输出：
     - data/battery_analysis_results.mat  — 整理后的结构化结果（后续脚本的输入）
     - figures/boxplots.png               — 循环寿命 / 总充电时间箱线图

  注意：本脚本是后续所有分析（模型对比、消融、SHAP、相关性）的数据预处理入口，
        请先运行本脚本生成 battery_analysis_results.mat，再运行其余脚本。
=============================================================================
"""

import h5py            # 读写 HDF5 格式（.mat v7.3 本质是 HDF5）
import numpy as np     # 数值计算
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端（服务器/无显示器环境下也能出图）
import matplotlib.pyplot as plt
from collections import defaultdict  # 用于按策略分组电池
import re               # 正则解析充电策略字符串
import os

# ============================================================
# 路径配置 (基于脚本自身位置，无论从哪里运行都正确)
# ============================================================
# __file__ 是本脚本的绝对路径；据此向上推导出项目根目录，避免硬编码路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 本脚本所在目录 code/
PROJ_DIR = os.path.dirname(BASE_DIR)                   # 项目根目录

MAT_FILE = os.path.join(PROJ_DIR, 'data', 'MIT-Stanford Battery Dataset_cleaned.mat')  # 原始数据
FIG_DIR  = os.path.join(PROJ_DIR, 'figures')           # 图片输出目录
OUT_MAT  = os.path.join(PROJ_DIR, 'data', 'battery_analysis_results.mat')  # 结果文件

os.makedirs(FIG_DIR, exist_ok=True)  # 确保图片目录存在（不存在则创建）

# 中文字体配置：优先 SimHei/微软雅黑，最后回退到 DejaVu Sans（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号（否则负号显示为方块）


def decode_policy(f, ref):
    """
    将 HDF5 中的 "policy_readable" 引用解码为可读字符串。

    MIT-Stanford 数据集中，字符串被存储为 HDF5 的变长引用或字符数组，
    这里逐字符取出其 ASCII 码再拼成字符串。

    参数:
        f   : h5py.File 对象（当前打开的数据文件）
        ref : HDF5 对象引用（指向 policy 字符串的存储位置）
    返回:
        str : 解码后的策略字符串，例如 "3.6C(80%)-4.6C"
    """
    obj = f[ref]                                   # 解引用，得到字符数组对象
    chars = [chr(int(x[0])) for x in obj[:]]       # 每个元素是 [ASCII码] 形式，转为字符
    return ''.join(chars)                          # 拼接成完整字符串


def parse_policy(policy_str):
    """
    从策略字符串中解析出三个充电参数。

    策略字符串形如 "3.6C(80%)-4.6C-newstructure"，含义：
       - 第一步以 C1=3.6C 倍率充电至 SOC=Q1=80%
       - 第二步以 C2=4.6C 倍率充电（至 80% 后进入 CV 阶段）
    其中 "-newstructure" 是数据集命名的后缀，需要先去除。

    参数:
        policy_str : 策略字符串
    返回:
        (C1, Q1, C2) : 三个浮点数；若格式不匹配则返回 (None, None, None)
    """
    s = policy_str.replace('-newstructure', '')  # 去掉后缀，简化正则匹配
    m = re.match(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C', s)  # 正则抓取三组数字
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None  # 匹配失败（异常策略）时返回 None


# ============================================================
# 步骤 1: 读取数据
# ============================================================
print('=' * 60)
print('步骤 1: 读取数据...')

f = h5py.File(MAT_FILE, 'r')        # 以只读方式打开原始数据集
bc = f['batch_combined']            # 顶层 group：包含每个电池的 summary/cycles 引用

N_BAT = 124                         # 数据集固定包含 124 个电池
batteries = []                      # 结果列表，每个元素是单个电池的字典

for i in range(N_BAT):
    # ---- 解析充电策略（policy_readable → 字符串 → 参数） ----
    ref_p = bc['policy_readable'][i, 0]   # 取出第 i 个电池的策略字符串引用
    policy_str = decode_policy(f, ref_p)  # 解码为字符串
    c1, q1, c2 = parse_policy(policy_str) # 解析出 C1/Q1/C2 三个参数

    # ---- 读取 summary 中的时间序列（容量、充电时间、内阻） ----
    ref_s = bc['summary'][i, 0]           # summary 数据集引用
    summary = f[ref_s]                    # 解引用

    n_rows = summary['QDischarge'].shape[1]   # 该电池的循环数 = 放电容量数组长度
    qd_raw = summary['QDischarge'][0, :].copy()  # 放电容量序列
    qc_raw = summary['QCharge'][0, :].copy()     # 充电容量序列
    ct_raw = summary['chargetime'][0, :].copy()  # 每循环充电时间序列
    ir_raw = summary['IR'][0, :].copy()          # 内阻序列

    # ---- 读取真实循环寿命标签（bc['cycle_life']，Severson 定义的 EOL 循环数） ----
    bl_ref = bc['cycle_life'][i, 0]       # cycle_life 引用
    true_cycle_life = f[bl_ref][0, 0]     # 解引用得到标量（真实 EOL 标签）

    # 前 5 个电池（i<=4）summary 中没有存储 cycle 字段，需手动生成 1..n
    # 其余电池直接读取 summary['cycle'] 的真实周期编号
    if i <= 4:
        cycle_numbers = np.arange(1, n_rows + 1, dtype=np.float64)
    else:
        cycle_numbers = summary['cycle'][0, :].copy()

    # ---- 派生特征 ----
    # SOH(健康状态) = 放电容量 / 标称容量 1.1 Ah（1.1 是该数据集电芯的额定容量）
    soh = qd_raw / 1.1
    # 循环寿命 = 真实 EOL 标签（bc['cycle_life']），不再用 QDischarge 数组长度作代理。
    # n_rows 是「记录循环数」，多数电芯会循环到 80% 阈值之后，故 n_rows 通常 >= 真实寿命。
    cycle_life = true_cycle_life
    n_recorded_cycles = n_rows
    # 总充电时间 = 各循环充电时间之和（小时）
    total_ct = np.sum(ct_raw)

    # 用字典存储该电池的全部信息
    batteries.append({
        'policy': policy_str,             # 策略字符串
        'C1': c1 if c1 else np.nan,       # 第一步倍率（None→NaN）
        'Q1': q1 if q1 else np.nan,       # SOC 切换点（None→NaN）
        'C2': c2 if c2 else np.nan,       # 第二步倍率（None→NaN）
        'cycle_life': cycle_life,         # 循环寿命（真实 EOL 标签）
        'n_recorded_cycles': n_recorded_cycles,  # 记录循环数（备查）
        'total_charge_time': total_ct,    # 总充电时间
        'QDischarge': qd_raw,             # 放电容量序列
        'SOH': soh,                       # 健康状态序列
        'chargetime': ct_raw,             # 充电时间序列
        'cycle_numbers': cycle_numbers,   # 周期编号
        'QCharge': qc_raw,                # 充电容量序列
        'IR': ir_raw,                     # 内阻序列
    })

f.close()  # 关闭文件释放资源
print(f'  读取完毕: {len(batteries)} 个电池')
print(f'  Bat 0 cycle: {batteries[0]["cycle_numbers"][0]:.0f} -> {batteries[0]["cycle_numbers"][-1]:.0f}')
print(f'  Bat 5 cycle: {batteries[5]["cycle_numbers"][0]:.0f} -> {batteries[5]["cycle_numbers"][-1]:.0f}')

# ============================================================
# 步骤 2: 统计指标
# ============================================================
print()
print('=' * 60)
print('步骤 2: 统计指标')

# 收集所有电池的循环寿命和总充电时间，用于整体分布统计
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

# 按策略字符串分组，统计每个策略下电池的循环寿命
policy_groups = defaultdict(list)
for b in batteries:
    policy_groups[b['policy']].append(b['cycle_life'])

# 计算每种策略的统计量（数量、均值、极值）
policy_stats = []
for pol, lifes in policy_groups.items():
    policy_stats.append({
        'policy': pol,
        'count': len(lifes),
        'mean_life': np.mean(lifes),
        'min_life': np.min(lifes),
        'max_life': np.max(lifes),
    })

# 按平均寿命降序排序（寿命最长的策略排前面）
policy_stats.sort(key=lambda x: x['mean_life'], reverse=True)

def get_params(pol_str):
    """策略字符串 → (C1, Q1, C2) 三个参数（供打印使用）"""
    c1, q1, c2 = parse_policy(pol_str)
    return c1, q1, c2

print()
print('  === 长寿策略 Top 5 ===')
for rank, ps in enumerate(policy_stats[:5], 1):  # 前 5（平均寿命最长）
    c1, q1, c2 = get_params(ps['policy'])
    print(f'  {rank}. "{ps["policy"]}"')
    print(f'     平均寿命: {ps["mean_life"]:.0f} cycles | '
          f'电池数: {ps["count"]} | '
          f'C1={c1}C, Q1={q1}%, C2={c2}C')

print()
print('  === 短寿策略 Top 5 ===')
for rank, ps in enumerate(policy_stats[-5:], 1):  # 后 5（平均寿命最短）
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

# 以写模式创建结果文件，组织为 battery_info / policy_stats 两个 group
f_out = h5py.File(OUT_MAT, 'w')
bi_grp = f_out.create_group('battery_info')  # 每个电池一个子 group

for i, b in enumerate(batteries):
    bg = bi_grp.create_group(f'bat_{i+1:03d}')  # 命名为 bat_001 ~ bat_124
    bg.create_dataset('policy', data=b['policy'].encode('utf-8'))  # 字符串需编码为字节
    # 注意：无效参数（NaN）统一存为 -1.0，作为"缺失"标记，便于下游判断
    bg.create_dataset('C1', data=b['C1'] if not np.isnan(b['C1']) else -1.0)
    bg.create_dataset('Q1', data=b['Q1'] if not np.isnan(b['Q1']) else -1.0)
    bg.create_dataset('C2', data=b['C2'] if not np.isnan(b['C2']) else -1.0)
    bg.create_dataset('cycle_life', data=b['cycle_life'])
    bg.create_dataset('n_recorded_cycles', data=b['n_recorded_cycles'])
    bg.create_dataset('total_charge_time', data=b['total_charge_time'])
    # 时间序列统一 reshape 成 (1, N) 行向量存储，与原始 summary 格式一致
    bg.create_dataset('cycle_numbers', data=b['cycle_numbers'].reshape(1, -1))
    bg.create_dataset('QDischarge', data=b['QDischarge'].reshape(1, -1))
    bg.create_dataset('QCharge', data=b['QCharge'].reshape(1, -1))
    bg.create_dataset('SOH', data=b['SOH'].reshape(1, -1))
    bg.create_dataset('chargetime', data=b['chargetime'].reshape(1, -1))
    bg.create_dataset('IR', data=b['IR'].reshape(1, -1))

# 策略统计结果单独存到一个 group（供相关性分析脚本直接读取）
ps_grp = f_out.create_group('policy_stats')
for i, ps in enumerate(policy_stats):
    pg = ps_grp.create_group(f'policy_{i+1:03d}')
    pg.create_dataset('policy', data=ps['policy'].encode('utf-8'))
    pg.create_dataset('count', data=ps['count'])
    pg.create_dataset('mean_life', data=ps['mean_life'])
    pg.create_dataset('min_life', data=ps['min_life'])
    pg.create_dataset('max_life', data=ps['max_life'])

# 顶层再存几个汇总标量/数组，方便后续直接读取
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

# 1×2 子图布局：左=循环寿命，右=总充电时间
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- 左图：循环寿命箱线图 ----
ax1 = axes[0]
bp1 = ax1.boxplot(cycle_lifes, widths=0.4, patch_artist=True,  # patch_artist 使箱体可填充颜色
                   medianprops={'color': 'black', 'linewidth': 2},   # 中位线样式
                   flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4})  # 离群点样式
bp1['boxes'][0].set_facecolor('#4ECDC4')   # 箱体填充色
ax1.set_ylabel('循环寿命')
ax1.set_title(f'循环寿命分布 (n={N_BAT})\n'
              f'最小值={cycle_lifes.min():.0f}, 最大值={cycle_lifes.max():.0f}, '
              f'中位数={np.median(cycle_lifes):.0f}')
ax1.set_xticks([])         # 单箱线图无分类，隐藏 x 轴刻度
ax1.grid(axis='y', alpha=0.3)

# ---- 右图：总充电时间箱线图 ----
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

# 以下断言用于校验数据读取的正确性，防止上游数据格式变化导致静默出错
assert len(batteries) == 124, f'电池数错误: {len(batteries)}'          # 电池总数
print(f'  OK 电池数 = {len(batteries)}')

# 校验 Bat 0 的周期编号范围（已知该电池有 1851 个循环）
assert batteries[0]['cycle_numbers'][0] == 1 and batteries[0]['cycle_numbers'][-1] == 1851
print(f'  OK Bat 0 cycle = 1->{int(batteries[0]["cycle_numbers"][-1])}')

# 校验 Bat 5（真实 cycle 字段的代表）从 1 开始
assert batteries[5]['cycle_numbers'][0] == 1
print(f'  OK Bat 5 cycle = 1->{int(batteries[5]["cycle_numbers"][-1])}')

# 校验 SOH 定义：SOH = QDischarge / 1.1
qd_sample = batteries[0]['QDischarge'][1]
soh_sample = batteries[0]['SOH'][1]
assert abs(soh_sample - qd_sample / 1.1) < 1e-10
print(f'  OK SOH = QDischarge / 1.1 (Qd={qd_sample:.4f}, SOH={soh_sample:.4f})')

# 校验策略分组覆盖全部 124 个电池
total_in_stats = sum(ps['count'] for ps in policy_stats)
assert total_in_stats == 124
print(f'  OK 策略分组覆盖 {total_in_stats} 个电池, 共 {len(policy_stats)} 种策略')

# 校验真实 EOL 标签 vs 记录循环数的差异（确认目标变量修正的幅度）
n_rec = np.array([b['n_recorded_cycles'] for b in batteries])
diffs = n_rec - cycle_lifes
n_diff = int(np.sum(diffs != 0))
print(f'  OK cycle_life 已采用真实 EOL 标签；{n_diff}/{N_BAT} 个电池的记录循环数 != 真实寿命')
print(f'     (记录循环数 - 真实寿命) 差值: min={diffs.min():.0f}, max={diffs.max():.0f}, '
      f'median={np.median(diffs):.0f}')

print()
print('=' * 60)
print('全部完成!')
print(f'  .mat: {OUT_MAT}')
print(f'  图片: {fig_path}')
