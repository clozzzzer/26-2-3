import pandas as pd
import numpy as np
import ast
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (
    confusion_matrix, silhouette_score,
    adjusted_rand_score, normalized_mutual_info_score,
    classification_report
)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 定义输出目录
FIG_DIR = './fig'
RESULT_DIR = './Result'

# 确保目录存在
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
    print(f"✅ 已创建图片保存目录: {FIG_DIR}")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# ===========================================

# 辅助函数：安全解析字符串列表 (用于 excess)
def safe_eval(x):
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '']: return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1]
    try:
        res = ast.literal_eval(x)
        if isinstance(res, tuple):
            return [res] if res else []
        elif isinstance(res, list):
            return res
        return []
    except Exception:
        return []


# 1. 读取数据
file_path = './data/new_feature_csv.csv'  # 保持与您运行成功的路径一致
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

# 清洗列名空格
df.columns = df.columns.str.strip()
print(f"检测到的列名: {df.columns.tolist()}")
print(f"数据形状: {df.shape}")

if 'Label' not in df.columns:
    raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")

# 2. 特征工程 (混合模式)
print("\n正在进行特征工程...")

feature_parts = []
feature_names_total = []

# --- A. 处理 excess 列 ---
if 'excess' in df.columns:
    print("检测到 'excess' 列，正在解析并转换...")
    df['excess_parsed'] = df['excess'].apply(safe_eval)

    mlb_excess = MultiLabelBinarizer()
    excess_features = mlb_excess.fit_transform(df['excess_parsed'])
    # 生成清晰的特征名，例如 ex_1_1_2
    excess_cols = [f"ex_{str(t).replace(', ', '_').replace('(', '').replace(')', '')}" for t in mlb_excess.classes_]

    feature_parts.append(excess_features)
    feature_names_total.extend(excess_cols)
    print(f"   -> excess 生成了 {len(excess_cols)} 个特征")
else:
    print("⚠️ 未找到 'excess' 列，跳过该步骤。")

# --- B. 直接提取其他数值列 ---
exclude_cols = {'Label', 'excess', 'excess_parsed'}
numeric_candidate_cols = [col for col in df.columns if col not in exclude_cols]

valid_numeric_cols = []
for col in numeric_candidate_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        valid_numeric_cols.append(col)
    else:
        print(f"⚠️  跳过非数值列: '{col}' (类型: {df[col].dtype})")

if valid_numeric_cols:
    print(f"检测到 {len(valid_numeric_cols)} 个现成数值特征。")
    numeric_data = df[valid_numeric_cols].values

    if np.isnan(numeric_data).any():
        print("⚠️  数值特征中存在缺失值，填充为 0...")
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)

    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)

# --- C. 合并所有特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X_raw = np.hstack(feature_parts)
y_true = df['Label']

print(f"\n✅ 特征矩阵构建完成:")
print(f"   总特征数: {X_raw.shape[1]}")
print(f"   样本数: {X_raw.shape[0]}")

# 3. 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 4. K-Means 聚类
n_clusters = y_true.nunique()
print(f"\n开始聚类 (K-Means), K = {n_clusters}...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X)

df['Cluster'] = cluster_labels

# 5. 评估指标
silhouette_avg = silhouette_score(X, cluster_labels)
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)

print(f"\n" + "=" * 40)
print("📊 聚类评估结果")
print("=" * 40)
print(f"轮廓系数 (Silhouette): {silhouette_avg:.4f}")
print(f"调整兰德指数 (ARI):    {ari:.4f}")
print(f"归一化互信息 (NMI):    {nmi:.4f}")

# --- 最佳映射准确率 ---
unique_true = sorted(y_true.unique(), key=str)
all_cluster_ids = list(range(n_clusters))

ct_int = pd.crosstab(y_true, cluster_labels)
ct_int = ct_int.reindex(index=unique_true, columns=all_cluster_ids, fill_value=0)
mat_int = ct_int.values

row_ind, col_ind = linear_sum_assignment(-mat_int)

best_map = {}
total_correct = 0
for r, c in zip(row_ind, col_ind):
    t_lab = unique_true[r]
    p_lab = all_cluster_ids[c]
    best_map[p_lab] = t_lab
    total_correct += mat_int[r, c]

accuracy_best = total_correct / len(df)
print(f"最佳映射准确率:        {accuracy_best:.4f} ({accuracy_best * 100:.2f}%)")

pred_mapped = []
for p in cluster_labels:
    if p in best_map:
        pred_mapped.append(best_map[p])
    else:
        pred_mapped.append(unique_true[0])  # 兜底

df['Best_Match_Pred'] = pred_mapped

print("\n📋 分类报告 (部分):")
print(classification_report(y_true, pred_mapped, digits=4, zero_division=0))

# ================= 绘图与保存部分 =================

# 准备绘图数据
pred_labels_plot = df['Cluster'].apply(lambda x: f"C{x}")
unique_pred_str = [f"C{x}" for x in all_cluster_ids]
unique_true_str = [str(x) for x in unique_true]

# --- 图 1: 混淆矩阵 ---
ct_plot = pd.crosstab(y_true, pred_labels_plot)
ct_plot = ct_plot.reindex(index=unique_true_str, columns=unique_pred_str, fill_value=0)

plt.figure(figsize=(14, 12))
sns.heatmap(ct_plot, annot=True, fmt='d', cmap='Blues', linewidths=.5, linecolor='gray')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy_best:.2%} | ARI: {ari:.3f}', fontsize=14)
plt.xlabel('Predicted Cluster', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

# 保存图 1
save_path_1 = os.path.join(FIG_DIR, 'KMeans_Confusion_Matrix.png')
plt.savefig(save_path_1, dpi=300)
print(f"✅ 图片已保存: {save_path_1}")
plt.show()

# --- 图 2: 簇特征均值分布 (修复空图问题) ---
# 关键修复：使用 X_raw 和 feature_names_total 构建临时 DataFrame
# 这样包含了 excess 解析出的所有特征，而不仅仅是原始 CSV 中的列
temp_df = pd.DataFrame(X_raw, columns=feature_names_total)
temp_df['Cluster'] = cluster_labels

if len(feature_names_total) > 0:
    # 计算方差，找出变化最大的前 5 个特征
    variances = temp_df[feature_names_total].var(axis=0)
    top_5_idx = variances.argsort()[::-1][:5]
    top_5_names = [feature_names_total[i] for i in top_5_idx]

    print(f"\n绘制特征分布图，选取方差最大的 5 个特征: {top_5_names}")

    group_means = temp_df.groupby('Cluster')[top_5_names].mean()

    plt.figure(figsize=(14, 8))
    # 绘制柱状图
    ax = group_means.plot(kind='bar', figsize=(14, 8), width=0.8)

    plt.title(f'Mean Values of Top 5 Variable Features per Cluster (Fig 2)', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Mean Value (Original Scale)', fontsize=12)
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图 2
    save_path_2 = os.path.join(FIG_DIR, 'KMeans_Feature_Distribution.png')
    plt.savefig(save_path_2, dpi=300)
    print(f"✅ 图片已保存: {save_path_2}")
    plt.show()
else:
    print("⚠️  没有特征可绘制分布图。")

# --- 图 3: 轮廓系数分布 (新增，替代原本可能为空的图) ---
# 如果之前的 Fig3 是空的，通常是因为逻辑错误。这里我们绘制一个更有意义的图：轮廓系数散点图
# 这能展示每个样本的聚类紧密度
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(X, cluster_labels)
y_lower = 10

plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    ith_cluster_silhouette_values = silhouette_vals[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.Spectral(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # 在簇中间标注簇号
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12, weight='bold')

    y_lower = y_upper + 10

# 垂直线表示平均轮廓系数
plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Avg Score: {silhouette_avg:.3f}')
plt.title(f'Silhouette Plot for Each Cluster (Fig 3)', fontsize=14)
plt.xlabel('Silhouette Coefficient Values')
plt.ylabel('Cluster')
plt.legend(loc="best")
plt.yticks([])  # 隐藏y轴刻度，因为已经手动标注了
plt.tight_layout()

# 保存图 3
save_path_3 = os.path.join(FIG_DIR, 'KMeans_Silhouette_Analysis.png')
plt.savefig(save_path_3, dpi=300)
print(f"✅ 图片已保存: {save_path_3}")
plt.show()

# ================= 结果保存 =================
# 保存详细 CSV
out_cols = ['Label', 'Cluster', 'Best_Match_Pred'] + valid_numeric_cols
save_df = df[['Label', 'Cluster', 'Best_Match_Pred']].copy()
save_df[valid_numeric_cols] = df[valid_numeric_cols]

output_file = os.path.join(RESULT_DIR, 'Kmeans_Result_Detail.csv')
save_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 结果详情已保存至: {output_file}")

# 保存统计摘要
summary_file = os.path.join(RESULT_DIR, 'Kmeans_Evaluation_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("K-Means Clustering Evaluation Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
    f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
    f.write(f"Normalized Mutual Info (NMI): {nmi:.4f}\n")
    f.write(f"Best Match Accuracy: {accuracy_best:.4f}\n\n")
    f.write(f"Total Features Used: {len(feature_names_total)}\n")
    f.write(f"Saved Figures:\n")
    f.write(f"  - {save_path_1}\n")
    f.write(f"  - {save_path_2}\n")
    f.write(f"  - {save_path_3}\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 程序全部执行完毕！请查看 ./fig 目录获取图片。")