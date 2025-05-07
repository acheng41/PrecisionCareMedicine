import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib import colors
import scipy.io as sio
from matplotlib.pyplot import figure
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sympy.printing.pretty.pretty_symbology import line_width

from getMatrixV3 import get_gait_parameters_insole3
from getMatrixV3 import reshape_insole_grid
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, stats, f_oneway

matplotlib.use('TkAgg')


#%%
def load_gait_data_stroke(walk,th):
    data = sio.loadmat(f'data/gait_recording_021925_walk{walk}.mat')
    insoleAll_l = data['insoleAll_l'].astype(np.float64)
    insoleAll_r = data['insoleAll_r'].astype(np.float64)
    t_insole_l = data['t_insole_l'].astype(np.float64)
    t_insole_r = data['t_insole_r'].astype(np.float64)

    # Remove borders
    insoleAll_l = insoleAll_l[1500:-1500, :]
    insoleAll_r = insoleAll_r[1500:-1500, :]
    t_insole_l = t_insole_l[1500:-1500, :]
    t_insole_r = t_insole_r[1500:-1500, :]

    gait = get_gait_parameters_insole3(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l, th)
    return gait

def load_gait_data_normal(walk,th):
    data = sio.loadmat(f'data/gait_recording_080624_walk{walk}.mat')
    insoleAll_l = data['insoleAll_l'].astype(np.float64)
    insoleAll_r = data['insoleAll_r'].astype(np.float64)
    t_insole_l = data['t_insole_l'].astype(np.float64)
    t_insole_r = data['t_insole_r'].astype(np.float64)

    # remove borders
    insoleAll_l = insoleAll_l[1500:-1500, :]
    insoleAll_r = insoleAll_r[1500:-1500, :]
    t_insole_l = t_insole_l[1500:-1500, :]
    t_insole_r = t_insole_r[1500:-1500, :]

    # get parameters
    gait = get_gait_parameters_insole3(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l, th)
    return gait

gait_1 = load_gait_data_stroke(4,  [50,50,50,50,3,3])
gait_2 = load_gait_data_stroke(2,  [100,100,50,50,2,2])
gait_3 = load_gait_data_normal(1,  [10,10,20,20,3,3])
gait_4 = load_gait_data_normal(2,  [10,10,10,10,3,3])

#%%
params = ['cycle_dur_r', 'stance_dur_r', 'stance_phase_r', 'cadence_r',
          'cycle_dur_l', 'stance_dur_l', 'stance_phase_l', 'cadence_l']
results = {}

for param in params:
    # extract data
    values_1 = np.array(gait_1[param])
    values_2 = np.array(gait_2[param])
    values_3 = np.array(gait_3[param])
    values_4 = np.array(gait_4[param])

    # compute mean and std
    mean_1, std_1 = np.mean(values_1), np.std(values_1, ddof=1)
    mean_2, std_2 = np.mean(values_2), np.std(values_2, ddof=1)
    mean_3, std_3 = np.mean(values_3), np.std(values_3, ddof=1)
    mean_4, std_4 = np.mean(values_4), np.std(values_4, ddof=1)

    # ANOVA
    F_statistic, p_value = stats.f_oneway(values_1, values_2)

    results[param] = {
        'walk1': f"{mean_1:.3f} ± {std_1:.3f}",
        'walk2': f"{mean_2:.3f} ± {std_2:.3f}",
        'walk3': f"{mean_3:.3f} ± {std_3:.3f}",
        'walk4': f"{mean_4:.3f} ± {std_4:.3f}",
        'ANOVA_F': F_statistic,
        'ANOVA_p': p_value
    }

    # If the ANOVA results showed significant differences, a Tukey HSD post hoc test was performed
    if p_value < 0.05:
        combined_values = np.concatenate([values_1, values_2])
        group_labels = ['walk1'] * len(values_1) + ['walk2'] * len(values_2)

        # Tukey HSD
        tukey_result = pairwise_tukeyhsd(combined_values, group_labels, alpha=0.05)

        # restore
        results[param]['Tukey HSD'] = tukey_result.summary()

# print
for param, res in results.items():
    print(f"parameter: {param}")
    print(f"  Walk1: {res['walk1']}")
    print(f"  Walk2: {res['walk2']}")
    print(f"  Walk3: {res['walk3']}")
    print(f"  Walk4: {res['walk4']}")
    print(f"  ANOVA F: {res['ANOVA_F']:.3f}, p-value: {res['ANOVA_p']:.5f}")

    # Prints
    if 'Tukey HSD' in res:
        print(f"  Tukey HSD: {res['Tukey HSD']}")

    print("-" * 50)

names = ['Right cycle duration', 'Right stance duration', 'Right stance phase', 'Right cadence',
         'Left cycle duration', 'Left stance duration', 'Left stance phase', 'Left cadence']
ylabels = ['s','s','%','/min','s','s','%','/min']
custom_palette = ["#de2d26", "#fc9272", "#3182bd", "#9ecae1"]
plt.figure(figsize=(10, 6))
for i, param in enumerate(params, 1):
    plt.subplot(2, 4, i)  # 创建子图
    sns.boxplot(data=[gait_1[param], gait_2[param], gait_3[param], gait_4[param]],
                palette=custom_palette)
    if i>4:
        plt.xticks([0, 1, 2, 3], ['stroke 0.5 0.5', 'stroke 0.75 0.75', 'healthy 0.6 0.6', 'healthy 0.6 1'], rotation=45)
    else:
        plt.xticks([0, 1, 2, 3], ['', '', '', ''])
    plt.title(names[i-1])
    plt.ylabel(ylabels[i-1])
plt.tight_layout()

#%% copVx,y and  cop_x y Comparison
gaits = [gait_1, gait_2, gait_3, gait_4]
colors = custom_palette
labels = ['stroke.0.5,0.5', 'stroke,0.75,0.75', 'healthy,0.6,0.6', 'healthy,0.6,1']

fig, axs = plt.subplots(1, 2, figsize=(8, 8))

# -------- left COP XY ----------
for i, gait in enumerate(gaits):
    x = gait['cop_x_avg_l']*2
    y = gait['cop_y_avg_l']*2
    std_x = gait['cop_x_std_l']*2
    std_y = gait['cop_y_std_l']*2

    axs[0].plot(x, y,  color=colors[i])
    axs[0].fill_betweenx(y, x - std_x, x + std_x, color=colors[i], alpha=0.2)

axs[0].set_title("Left COP Trajectory")
# axs[0].legend()
axs[0].axis('equal')
axs[0].set_ylabel('Posterior <--> Anterior (mm)')
axs[0].set_xlabel('Lateral <--> Medial (mm)')

# -------- right COP XY ----------
for i, gait in enumerate(gaits):
    x = (16-gait['cop_x_avg_r'])*2
    y = gait['cop_y_avg_r']*2
    std_x = gait['cop_x_std_r']*2
    std_y = gait['cop_y_std_r']*2
    axs[1].plot(x, y, color=colors[i])
    axs[1].fill_betweenx(y, x - std_x, x + std_x, color=colors[i], alpha=0.2)

axs[1].set_title("Right COP Trajectory")
axs[1].legend()
axs[1].axis('equal')
axs[1].set_ylabel('Posterior <--> Anterior (mm)')
axs[1].set_xlabel('Medial <--> Lateral (mm)')
plt.tight_layout()


fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

vars = [
    ('cop_vx_avg_l', 'cop_vx_std_l', 'Left CoP Velocity X'),
    ('cop_vx_avg_r', 'cop_vx_std_r', 'Right CoP Velocity X'),
    ('cop_vy_avg_l', 'cop_vy_std_l', 'Left CoP Velocity Y'),
    ('cop_vy_avg_r', 'cop_vy_std_r', 'Instantaneous CoP Velocity (AP-y)'),
]

x = np.linspace(0, 100, len(gaits[0]['cop_vx_avg_l']))  # 横坐标

for idx, (mean_key, std_key, title) in enumerate(vars):
    ax = axs[idx]
    for i, gait in enumerate(gaits):
        mean = gait[mean_key]
        # std = gait[std_key]
        ax.plot(x, -mean, label=labels[i], color=colors[i])
        # ax.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Stance Phase (%)')
    ax.set_ylabel('Velocity (mm/s, forward positive)')
    ax.tick_params()


ax.legend(title='Subject, Left Speed, Right Speed(m/s)', ncol=2)
plt.tight_layout()

#%%
params = ['cop_range_x_l',  'cop_range_y_l', 'cop_length_l', 'cop_range_x_r', 'cop_range_y_r',  'cop_length_r']
results = {}

for param in params:

    values_1 = np.array(gait_1[param])
    values_2 = np.array(gait_2[param])
    values_3 = np.array(gait_3[param])
    values_4 = np.array(gait_4[param])

    mean_1, std_1 = np.mean(values_1), np.std(values_1, ddof=1)
    mean_2, std_2 = np.mean(values_2), np.std(values_2, ddof=1)
    mean_3, std_3 = np.mean(values_3), np.std(values_3, ddof=1)
    mean_4, std_4 = np.mean(values_4), np.std(values_4, ddof=1)

    # ANOVA
    F_statistic, p_value = stats.f_oneway(values_1, values_2, values_3, values_4)

    # restore results
    results[param] = {
        'walk1': f"{mean_1:.3f} ± {std_1:.3f}",
        'walk2': f"{mean_2:.3f} ± {std_2:.3f}",
        'walk3': f"{mean_3:.3f} ± {std_3:.3f}",
        'walk4': f"{mean_4:.3f} ± {std_4:.3f}",
        'ANOVA_F': F_statistic,
        'ANOVA_p': p_value
    }

    # Tukey HSD
    if p_value < 0.05:
        combined_values = np.concatenate([values_1, values_2, values_3, values_4])
        group_labels = ['walk1'] * len(values_1) + ['walk2'] * len(values_2) + ['walk3'] * len(values_3) + ['walk4'] * len(values_4)

        tukey_result = pairwise_tukeyhsd(combined_values, group_labels, alpha=0.05)

        results[param]['Tukey HSD'] = tukey_result.summary()

# print results
for param, res in results.items():
    print(f"Parameter: {param}")
    print(f"  Walk1: {res['walk1']}")
    print(f"  Walk2: {res['walk2']}")
    print(f"  Walk3: {res['walk3']}")
    print(f"  Walk4: {res['walk4']}")
    print(f"  ANOVA F: {res['ANOVA_F']:.3f}, p-value: {res['ANOVA_p']:.5f}")

    if 'Tukey HSD' in res:
        print(f"  Tukey HSD: {res['Tukey HSD']}")

    print("-" * 50)

names = ['left cop range x', 'left cop range y'  , 'left cop length', 'Right CoP Range (ML-x)', 'CoP Range (AP-y)',
          'CoP Path Length']
ylabels = ['mm','mm','mm','mm','mm','mm']
custom_palette = ["#de2d26", "#fc9272", "#3182bd", "#9ecae1"]
plt.figure(figsize=(20, 4))
for i, param in enumerate(params, 1):
    ax = plt.subplot(2, 3, i)  # 创建子图
    data = [gait_1[param], gait_2[param], gait_3[param], gait_4[param]]
    box = sns.boxplot(data=data, palette=custom_palette, ax=ax)

    # 设置x轴标签
    if i >= 4:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['stroke 0.5', 'stroke 0.75', 'healthy 0.6', 'healthy 1'], rotation=15,fontsize=8)
    else:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['', '', '', ''], rotation=15,fontsize=8)

    ax.set_title(names[i-1],fontsize=10)
    ax.set_ylabel(ylabels[i-1],fontsize=8)
    ax.tick_params(size=8)

    # --- 显著性标注部分 ---
    # if 'Tukey HSD' in results[param]:
    #     tukey_table = results[param]['Tukey HSD'].data[1:]  # 去掉表头
    #     max_y = max([max(d) for d in data])  # 找最大y值
    #     h = (max_y * 0.05)  # 每层显著性线高度间隔
    #
    #     sig_level = 0  # 用于逐层叠加线条高度
    #     for row in tukey_table:
    #         group1, group2, meandiff, p_adj, lower, upper, reject = row
    #         if reject:  # 如果显著
    #             # 将 group1/group2 转换为索引（0~3）
    #             gmap = {'walk1': 0, 'walk2': 1, 'walk3': 2, 'walk4': 3}
    #             x1, x2 = gmap[group1], gmap[group2]
    #             y = max_y + h * (sig_level + 1)
    #             ax.plot([x1, x1, x2, x2], [y, y+0.2*h, y+0.2*h, y], lw=1.5, color='k')
    #             ax.text((x1 + x2) / 2, y + 0.3*h, '*', ha='center', va='bottom', color='k', fontsize=14)
    #             sig_level += 1
    #     # 获取当前y轴范围
    #     ymin, ymax = ax.get_ylim()
    #     # 设定新的最大y轴：当前最大值+一部分空间
    #     ax.set_ylim(top=ymax + h * 3)

plt.tight_layout()
plt.show()