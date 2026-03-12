import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from collections import Counter

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 定义输出文件夹
output_dir = 'fig'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建文件夹: {output_dir}")


def safe_eval(x):
    """安全解析字符串为列表/元组"""
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '', 'nan']: return []
        # 处理可能的引号包裹
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
    """保存高清图片"""
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"已保存: {path}")


# ================= 1. 读取数据 =================
file_path = './data/new_feature_csv.csv'
if not os.path.exists(file_path):
    file_path = 'new_feature_csv.csv'

print(f"正在读取文件: {file_path} ...")
try:
    df = pd.read_csv(file_path)
    print("文件读取成功！")
except Exception as e:
    print(f"读取失败: {e}")
    raise e

# 检查必要列
required_cols = ['Label', 'excess', '6_Count', 'Lack_Count']
lk_cols = [f'lk_elem_{i}' for i in range(1, 6)]
required_cols.extend(lk_cols)

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"错误：新CSV文件中缺少以下关键列，请确认文件生成是否正确: {missing_cols}")

print(f"检测到列: {df.columns.tolist()}")
print(f"样本总数: {len(df)}")
print(f"设备类别: {df['Label'].unique()}")

# ================= 2. 数据预处理 =================
# 解析 excess 列 (保持原有逻辑)
df['excess_parsed'] = df['excess'].apply(safe_eval)

# 【修改点】：不再解析 lack 列，也不再进行 lack 相关的计算
# 直接使用现有的 lk_elem_1 ~ lk_elem_5 和 Lack_Count 列
print("直接使用已有的 lk_elem 和 Lack_Count 特征，跳过重复计算。")

# 统计每个 Label 的数量
label_counts = df['Label'].value_counts()
labels = label_counts.index.tolist()

# 自动选择调色板
n_labels = len(labels)
if n_labels <= 10:
    palette = sns.color_palette("tab10", n_labels)
elif n_labels <= 20:
    palette = sns.color_palette("tab20", n_labels)
else:
    palette = sns.color_palette("husl", n_labels)

label_color_map = {label: palette[i] for i, label in enumerate(labels)}
df['color'] = df['Label'].map(label_color_map)

# ================= 3. 可视化绘图 =================

# --- Fig 1: 6_Count 分布 ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x=df.index, y='6_Count', hue='Label', palette=label_color_map, legend=False)
plt.title('Distribution of 6_Count per Sample')
plt.xlabel('Sample Index')
plt.ylabel('Count of 6')
plt.xticks([])  # 样本太多，隐藏x轴刻度
save_figure(plt.gcf(), 'Visual_6_Count_Distribution.png')
plt.close()

# --- Fig 2: Excess 模式热力图 (Top 15) ---
# 展平所有 excess 模式并计数
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
    plt.xlabel('Excess Pattern')
    plt.ylabel('Label')
    save_figure(plt.gcf(), 'Visual_Excess_Heatmap_Top15.png')
    plt.close()
else:
    print("未找到 excess 模式，跳过 Fig 2。")

# --- Fig 3: Lack 元素散点图 (基于现有的 lk_elem 列) ---
# 重构数据用于绘图：将宽表转为长表
lack_plot_data = []
for idx, row in df.iterrows():
    for i in range(1, 6):
        col_name = f'lk_elem_{i}'
        val = row[col_name]
        if val == 1:  # 只画缺失的点
            lack_plot_data.append({
                'Sample_Index': idx,
                'Missing_Element': i,
                'Label': row['Label'],
                'Color': row['color']
            })

lack_df = pd.DataFrame(lack_plot_data)

if not lack_df.empty:
    plt.figure(figsize=(12, 6))
    # 使用 scatterplot，jitter 防止点重叠
    sns.scatterplot(data=lack_df, x='Sample_Index', y='Missing_Element',
                    hue='Label', palette=label_color_map, s=100, alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.title('Missing Elements Distribution (lk_elem features)')
    plt.xlabel('Sample Index')
    plt.ylabel('Missing Element (1-5)')
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_figure(plt.gcf(), 'Visual_Lack_Elements_Scatter.png')
    plt.close()
else:
    print("未发现任何缺失元素数据，跳过 Fig 3。")

# --- Fig 4: Lack_Count 分布 ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x=df.index, y='Lack_Count', hue='Label', palette=label_color_map, legend=False)
plt.title('Distribution of Lack_Count (Total Missing Elements) per Sample')
plt.xlabel('Sample Index')
plt.ylabel('Count of Missing Elements')
plt.xticks([])
save_figure(plt.gcf(), 'Visual_Lack_Count_Distribution.png')
plt.close()

# --- Fig 5: Excess 样本分布矩阵 (简化版) ---
if top_15_patterns:
    plt.figure(figsize=(15, 8))
    # 创建一个映射，将每个样本的 top15 模式存在与否标记出来
    matrix_data = np.zeros((len(df), len(top_15_patterns)))

    for idx, row in df.iterrows():
        patterns = [str(x) for x in row['excess_parsed']]
        for j, p in enumerate(top_15_patterns):
            if p in patterns:
                matrix_data[idx, j] = 1

    # 绘制图像，每一行是一个样本，每一列是一个模式
    # 为了视觉效果，我们只画存在的点
    y_indices, x_indices = np.where(matrix_data == 1)

    colors_row = df.iloc[y_indices]['color']

    plt.scatter(x_indices, y_indices, c=colors_row, s=10, alpha=0.8)
    plt.xticks(range(len(top_15_patterns)), [p[:15] + '...' if len(p) > 15 else p for p in top_15_patterns],
               rotation=45, ha='right')
    plt.yticks([])
    plt.xlabel('Top 15 Excess Patterns')
    plt.ylabel('Sample Index')
    plt.title('Sample Distribution Matrix for Top 15 Excess Patterns')

    # 添加图例代理
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in palette]
    plt.legend(handles, labels, title="Label", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    save_figure(plt.gcf(), 'Visual_Excess_Matrix_Distribution.png')
    plt.close()

# --- Fig 6, 7, 8: Range 系列特征 (如果存在) ---
range_cols = ['Range_1', 'Range_2', 'Range_1_2']
existing_range_cols = [col for col in range_cols if col in df.columns]

for col in existing_range_cols:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=df.index, y=col, hue='Label', palette=label_color_map, legend=False)
    plt.title(f'Distribution of {col} per Sample')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.xticks([])
    save_figure(plt.gcf(), f'Visual_{col}_Distribution.png')
    plt.close()

if len(existing_range_cols) < len(range_cols):
    print(f"警告：部分 Range 列未找到 ({set(range_cols) - set(existing_range_cols)})，跳过对应绘图。")

print("\n✅ 所有可视化任务完成！图片已保存至 '{}' 文件夹。".format(output_dir))