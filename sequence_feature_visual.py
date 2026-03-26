import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from collections import Counter

# ================= 配置区域 =================
file_path = 'data/feature/csv0419_1/All_feature_csv0419_1.csv'
if not os.path.exists(file_path):
    file_path = 'new_feature_csv.csv'
if not os.path.exists(file_path):
    file_path = 'feature_csv_1_with_new_features.csv'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

output_dir = 'fig'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def safe_eval(x):
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '', 'nan']: return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1]
    try:
        res = ast.literal_eval(x)
        if isinstance(res, tuple):
            return [res] if res else []
        elif isinstance(res, list):
            return res
        return []
    except:
        return []


def save_figure(fig, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"已保存: {path}")


# ================= 1. 读取数据 =================
print(f"正在读取文件: {file_path} ...")
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 去除空格
    print("文件读取成功！")
except Exception as e:
    print(f"读取失败: {e}")
    raise e

print(f"可用列名: {df.columns.tolist()}")

# 【修正】严格匹配小写列名
target_col = 'lk_count'
if target_col not in df.columns:
    # 尝试寻找最接近的
    candidates = [c for c in df.columns if 'lk' in c.lower() and 'count' in c.lower()]
    if candidates:
        target_col = candidates[0]
        print(f"⚠️ 未找到 'lk_count'，自动切换为: {target_col}")
    else:
        raise ValueError(f"错误：找不到包含 'lk' 和 'count' 的列。当前列: {df.columns.tolist()}")

print(f"使用列 '{target_col}' 进行绘图。样本数: {len(df)}")

# 预处理
df['excess_parsed'] = df['excess'].apply(safe_eval)
labels = df['Label'].unique()
n_labels = len(labels)

# 调色板
if n_labels <= 10:
    palette = sns.color_palette("tab10", n_labels)
elif n_labels <= 20:
    palette = sns.color_palette("tab20", n_labels)
else:
    palette = sns.color_palette("husl", n_labels)
label_color_map = {label: palette[i] for i, label in enumerate(labels)}

# ================= 2. 可视化绘图 (针对 700+ 样本优化) =================

# --- Fig 1: Excess 热力图 (保持不变) ---
print("\n生成 Fig 1: Excess Heatmap...")
all_excess_patterns = []
for lst in df['excess_parsed']:
    all_excess_patterns.extend([str(x) for x in lst])
counter = Counter(all_excess_patterns)
top_15_patterns = [item[0] for item in counter.most_common(15)]

if top_15_patterns:
    heatmap_data = pd.DataFrame(index=labels, columns=top_15_patterns, data=0)
    for _, row in df.iterrows():
        lb = row['Label']
        patterns = [str(x) for x in row['excess_parsed']]
        for p in patterns:
            if p in top_15_patterns:
                heatmap_data.loc[lb, p] += 1

    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    plt.title('Top 15 Excess Patterns Frequency by Label')
    save_figure(plt.gcf(), '01_Excess_Heatmap.png')
    plt.close()

# --- Fig 2: lk_count 分布 [优化版：箱线图] ---
# 解决 700 样本过于密集的问题：使用箱线图展示统计分布
print("生成 Fig 2: lk_count Distribution (Boxplot)...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Label', y=target_col, palette=palette, showfliers=True)
sns.swarmplot(data=df, x='Label', y=target_col, color=".25", size=3, alpha=0.5)  # 叠加少量散点显示密度
plt.title(f'Distribution of {target_col} by Label (Boxplot with Outliers)\nTotal Samples: {len(df)}', fontsize=14)
plt.xlabel('Label')
plt.ylabel(f'{target_col} Value')
plt.grid(axis='y', linestyle='--', alpha=0.5)
save_figure(plt.gcf(), '02_lk_count_Boxplot.png')
plt.close()

# --- Fig 3: lk_count 分布 [优化版：分面条形图] ---
# 如果您必须看每个样本的具体值，我们将它们按 Label 拆分到子图中
print("生成 Fig 3: lk_count Distribution (Faceted Bars)...")
num_cols = 2
num_rows = (n_labels + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]
    subset = df[df['Label'] == label].reset_index(drop=True)

    # 绘制条形图
    ax.bar(subset.index, subset[target_col], color=label_color_map[label], edgecolor='white', width=0.8)
    ax.set_title(f'Label: {label} (Count: {len(subset)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Index (within group)')
    ax.set_ylabel(target_col)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 隐藏多余的子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f'Sample-wise {target_col} Distribution (Split by Label)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_figure(fig, '03_lk_count_Faceted_Bars.png')
plt.close()

# --- Fig 4: lk_count 趋势图 [优化版：滚动平均] ---
# 展示随样本索引变化的宏观趋势，过滤噪点
print("生成 Fig 4: lk_count Trend (Rolling Mean)...")
plt.figure(figsize=(14, 6))

# 按 Label 分别画趋势线
for label in labels:
    subset = df[df['Label'] == label].sort_index()  # 确保按原始顺序
    if len(subset) < 10:
        # 数据太少直接画点
        plt.plot(subset.index, subset[target_col], 'o-', label=label, color=label_color_map[label], markersize=4,
                 alpha=0.6)
    else:
        # 数据多则画滚动平均 (窗口大小设为样本数的 5% 或固定 20)
        window = max(5, int(len(subset) * 0.05))
        rolling_mean = subset[target_col].rolling(window=window, center=True).mean()

        plt.plot(subset.index, rolling_mean, '-', label=f'{label} (Rolling Avg)', color=label_color_map[label],
                 linewidth=2.5)
        # 可选：淡淡地画出原始数据背景
        # plt.plot(subset.index, subset[target_col], '.', color=label_color_map[label], alpha=0.1, markersize=1)

plt.title(f'Macro Trend of {target_col} (Rolling Mean Window={window})', fontsize=14)
plt.xlabel('Global Sample Index')
plt.ylabel(f'{target_col} (Smoothed)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
save_figure(plt.gcf(), '04_lk_count_Trend.png')
plt.close()

# --- Fig 5: Lack Elements Scatter (保持逻辑，适配新列名) ---
print("生成 Fig 5: Lack Elements Scatter...")
lk_cols = []
for i in range(1, 6):
    # 尝试匹配 lk_elem_1, lk_elem1, lack_1 等
    candidates = [c for c in df.columns if
                  str(i) in c and ('lk' in c.lower() or 'lack' in c.lower()) and 'count' not in c.lower()]
    if candidates:
        lk_cols.append(candidates[0])

if lk_cols:
    plot_data = []
    for idx, row in df.iterrows():
        for col in lk_cols:
            if row[col] == 1:  # 假设 1 表示缺失
                try:
                    elem_id = int(''.join(filter(str.isdigit, col)))
                except:
                    elem_id = 0
                plot_data.append(
                    {'idx': idx, 'elem': elem_id, 'label': row['Label'], 'color': label_color_map[row['Label']]})

    if plot_data:
        pdf = pd.DataFrame(plot_data)
        plt.figure(figsize=(14, 6))
        sns.scatterplot(data=pdf, x='idx', y='elem', hue='label', palette=label_color_map, s=60, edgecolor='w',
                        alpha=0.8)
        plt.title('Missing Elements Distribution (Scatter)')
        plt.xlabel('Sample Index')
        plt.ylabel('Element ID')
        plt.yticks(sorted(pdf['elem'].unique()))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save_figure(plt.gcf(), '05_Lack_Elements_Scatter.png')
        plt.close()
else:
    print("未找到 lack element 列，跳过散点图。")

# --- Fig 6: Excess Matrix (保持逻辑) ---
# ... (此处省略重复代码，逻辑同前，仅文件名更新) ...
if top_15_patterns:
    print("生成 Fig 6: Excess Matrix...")
    plt.figure(figsize=(15, 8))
    matrix_data = np.zeros((len(df), len(top_15_patterns)))
    for idx, row in df.iterrows():
        patterns = [str(x) for x in row['excess_parsed']]
        for j, p in enumerate(top_15_patterns):
            if p in patterns:
                matrix_data[idx, j] = 1

    y_idx, x_idx = np.where(matrix_data == 1)
    if len(y_idx) > 0:
        colors = df.iloc[y_idx]['Label'].map(label_color_map)
        plt.scatter(x_idx, y_idx, c=colors, s=15, alpha=0.8, edgecolors='none')
        plt.xticks(range(len(top_15_patterns)), [p[:12] + '..' if len(p) > 12 else p for p in top_15_patterns],
                   rotation=45, ha='right')
        plt.yticks([])
        plt.xlabel('Top Patterns')
        plt.ylabel('Sample Index')
        plt.title('Excess Pattern Matrix')
        # 添加图例略
        save_figure(plt.gcf(), '06_Excess_Matrix.png')
    plt.close()

print("\n✅ 所有优化后的图表已生成！请查看 ./fig 文件夹。")
print("💡 提示：")
print("   - 02_...Boxplot.png : 最适合快速对比各类别分布统计。")
print("   - 03_...Faceted.png : 适合查看每个样本的具体数值（已拆分）。")
print("   - 04_...Trend.png   : 适合查看整体变化趋势。")