import pandas as pd
import numpy as np
import ast
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, classification_report
)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

FIG_DIR = './fig'
RESULT_DIR = './Result'
RF_IMPORTANCE_PATH = './Result/RF_Importance.csv'

# 确保目录存在
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# ===========================================

# 辅助函数：安全解析字符串列表
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
file_path = './data/new_feature_csv.csv'
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

df.columns = df.columns.str.strip()
print(f"检测到的列名: {df.columns.tolist()}")
print(f"数据形状: {df.shape}")

if 'Label' not in df.columns:
    raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")

# 2. 加载 RF 特征重要性权重
print(f"\n正在加载 RF 特征重要性权重: {RF_IMPORTANCE_PATH} ...")
if not os.path.exists(RF_IMPORTANCE_PATH):
    raise FileNotFoundError(f"❌ 错误：找不到文件 {RF_IMPORTANCE_PATH}，请先运行 RF.py 生成该文件。")

rf_imp_df = pd.read_csv(RF_IMPORTANCE_PATH)
# 确保列名正确 (假设 CSV 列为 'Feature_Name', 'Importance_Score')
if 'Feature_Name' not in rf_imp_df.columns or 'Importance_Score' not in rf_imp_df.columns:
    # 尝试兼容可能的列名变化
    cols = rf_imp_df.columns
    if len(cols) >= 2:
        rf_imp_df = rf_imp_df.rename(columns={cols[0]: 'Feature_Name', cols[1]: 'Importance_Score'})
    else:
        raise ValueError("❌ RF 重要性文件格式不正确，必须包含特征名和权重列。")

# 创建权重字典
weight_map = pd.Series(rf_imp_df['Importance_Score'].values, index=rf_imp_df['Feature_Name']).to_dict()
print(f"✅ 已加载 {len(weight_map)} 个特征的权重。")
print(f"   最大权重: {max(weight_map.values()):.4f}, 最小权重: {min(weight_map.values()):.4f}")

# 3. 特征工程 (必须与 RF.py 完全一致)
print("\n正在进行特征工程...")

feature_parts = []
feature_names_total = []

# --- A. Excess 特征 ---
if 'excess' in df.columns:
    df['excess_parsed'] = df['excess'].apply(safe_eval)
    mlb_excess = MultiLabelBinarizer()
    excess_features = mlb_excess.fit_transform(df['excess_parsed'])
    excess_cols = [f"ex_{str(t).replace(', ', '_').replace('(', '').replace(')', '')}" for t in mlb_excess.classes_]

    feature_parts.append(excess_features)
    feature_names_total.extend(excess_cols)
else:
    print("⚠️ 未找到 'excess' 列。")

# --- B. 数值特征 ---
exclude_cols = {'Label', 'excess', 'excess_parsed'}
numeric_candidate_cols = [col for col in df.columns if col not in exclude_cols]

valid_numeric_cols = []
for col in numeric_candidate_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        valid_numeric_cols.append(col)

if valid_numeric_cols:
    numeric_data = df[valid_numeric_cols].values
    if np.isnan(numeric_data).any():
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)

    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)

# --- C. 合并特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X_raw = np.hstack(feature_parts)
y_true = df['Label']

print(f"\n✅ 特征矩阵构建完成:")
print(f"   总特征数: {X_raw.shape[1]}")
print(f"   样本数: {X_raw.shape[0]}")

# ================= 关键步骤：应用 RF 权重 =================
print("\n⚖️ 正在应用 RF 特征权重进行加权...")

# 1. 检查特征对齐
missing_features = set(feature_names_total) - set(weight_map.keys())
extra_features = set(weight_map.keys()) - set(feature_names_total)

if missing_features:
    print(f"⚠️  警告：以下 {len(missing_features)} 个特征在 RF 重要性表中缺失，将赋予平均权重或0权重：")
    # print(list(missing_features)[:5], "...")
    # 策略：对于缺失权重的特征，赋予所有现有权重的平均值，避免直接丢弃或设为0导致信息丢失
    avg_weight = np.mean(list(weight_map.values()))
    for feat in missing_features:
        weight_map[feat] = avg_weight

# 2. 构建权重向量 (顺序必须与 feature_names_total 严格一致)
weight_vector = np.array([weight_map[feat] for feat in feature_names_total])

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 4. 加权 (Element-wise Multiplication)
# X_weighted = X_scaled * weight_vector
# 注意：weight_vector 需要广播到 (N_samples, N_features)
X_weighted = X_scaled * weight_vector

print(f"   加权完成。前 5 个特征的权重: {weight_vector[:5]}")
print(f"   加权后数据形状: {X_weighted.shape}")

# =======================================================

# 4. K-Means 聚类 (使用加权后的数据)
n_clusters = y_true.nunique()
print(f"\n开始加权 K-Means 聚类, K = {n_clusters}...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_weighted)  # 使用 X_weighted 而不是 X_scaled

df['Cluster'] = cluster_labels

# 5. 评估指标
silhouette_avg = silhouette_score(X_weighted, cluster_labels)  # 在加权空间计算
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)

print(f"\n" + "=" * 50)
print("📊 加权聚类评估结果 (RF-Weighted K-Means)")
print("=" * 50)
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
        pred_mapped.append(unique_true[0])

df['Best_Match_Pred'] = pred_mapped

print("\n📋 分类报告 (部分):")
print(classification_report(y_true, pred_mapped, digits=4, zero_division=0))

# ================= 绘图与保存部分 =================

pred_labels_plot = df['Cluster'].apply(lambda x: f"C{x}")
unique_pred_str = [f"C{x}" for x in all_cluster_ids]
unique_true_str = [str(x) for x in unique_true]

# --- 图 1: 混淆矩阵 ---
ct_plot = pd.crosstab(y_true, pred_labels_plot)
ct_plot = ct_plot.reindex(index=unique_true_str, columns=unique_pred_str, fill_value=0)

plt.figure(figsize=(14, 12))
sns.heatmap(ct_plot, annot=True, fmt='d', cmap='Blues', linewidths=.5, linecolor='gray')
plt.title(f'Weighted Confusion Matrix (RF-KMeans)\nAcc: {accuracy_best:.2%} | ARI: {ari:.3f}', fontsize=14)
plt.xlabel('Predicted Cluster', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

save_path_1 = os.path.join(FIG_DIR, 'KMeans_Weighted_Confusion_Matrix.png')
plt.savefig(save_path_1, dpi=300)
print(f"✅ 图片已保存: {save_path_1}")
plt.show()

# --- 图 2: 加权特征分布 (使用原始数据的均值，但按重要性排序) ---
temp_df = pd.DataFrame(X_raw, columns=feature_names_total)
temp_df['Cluster'] = cluster_labels

# 按 RF 重要性排序特征，取前 5 个最重要的画
sorted_feats = sorted(feature_names_total, key=lambda x: weight_map.get(x, 0), reverse=True)
top_5_weighted_feats = sorted_feats[:5]

print(f"\n绘制 Top 5 重要特征分布: {top_5_weighted_feats}")

group_means = temp_df.groupby('Cluster')[top_5_weighted_feats].mean()

plt.figure(figsize=(14, 8))
ax = group_means.plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title(f'Mean Values of Top 5 RF-Important Features per Cluster (Fig 2)', fontsize=14)
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('Mean Value (Original Scale)', fontsize=12)
plt.legend(title='Features (Sorted by RF Weight)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path_2 = os.path.join(FIG_DIR, 'KMEans_Weighted_Feature_Distribution.png')
plt.savefig(save_path_2, dpi=300)
print(f"✅ 图片已保存: {save_path_2}")
plt.show()

# --- 图 3: 轮廓系数分析 (基于加权空间) ---
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(X_weighted, cluster_labels)
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

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12, weight='bold')
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Avg Score: {silhouette_avg:.3f}')
plt.title(f'Silhouette Plot (RF-Weighted Space) (Fig 3)', fontsize=14)
plt.xlabel('Silhouette Coefficient Values')
plt.ylabel('Cluster')
plt.legend(loc="best")
plt.yticks([])
plt.tight_layout()

save_path_3 = os.path.join(FIG_DIR, 'KMeans_Weighted_Silhouette_Analysis.png')
plt.savefig(save_path_3, dpi=300)
print(f"✅ 图片已保存: {save_path_3}")
plt.show()

# ================= 结果保存 =================
out_cols = ['Label', 'Cluster', 'Best_Match_Pred'] + valid_numeric_cols
save_df = df[['Label', 'Cluster', 'Best_Match_Pred']].copy()
save_df[valid_numeric_cols] = df[valid_numeric_cols]

output_file = os.path.join(RESULT_DIR, 'Kmeans_Weighted_Result_Detail.csv')
save_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 加权聚类结果详情已保存至: {output_file}")

# 保存对比摘要
summary_file = os.path.join(RESULT_DIR, 'Kmeans_Weighted_Evaluation_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("RF-Weighted K-Means Clustering Evaluation Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
    f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
    f.write(f"Normalized Mutual Info (NMI): {nmi:.4f}\n")
    f.write(f"Best Match Accuracy: {accuracy_best:.4f}\n\n")
    f.write(f"Weighting Strategy: Scaled Features * RF_Importance\n")
    f.write(f"Total Features Used: {len(feature_names_total)}\n")
    f.write(f"Saved Figures:\n")
    f.write(f"  - {save_path_1}\n")
    f.write(f"  - {save_path_2}\n")
    f.write(f"  - {save_path_3}\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 加权聚类程序全部执行完毕！请查看 ./fig 目录获取图片。")