import pandas as pd
import numpy as np
import ast
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

FIG_DIR = './fig'
RESULT_DIR = './Result'
IMPORTANCE_FILE = './Result/RF_All_Importance.csv'  # 注意路径调整为 Result 目录

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# ===========================================

def clean_feature_name(name):
    s = str(name)
    s = s.replace(', ', '_')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace("'", '')
    s = s.replace('"', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    return s


def safe_eval_to_str_list(x):
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '', 'nan', 'None']: return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1]

    parsed_list = []
    try:
        res = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(res, tuple):
            parsed_list = [res]
        elif isinstance(res, list):
            parsed_list = res
        else:
            parsed_list = [res] if res is not None else []
    except Exception:
        if isinstance(x, str):
            parsed_list = [item.strip() for item in x.split(',') if item.strip()]
        else:
            parsed_list = [x] if x is not None else []
    return [str(item) for item in parsed_list]


# ================= 1. 读取外部特征重要性文件 =================
print(f"📄 正在读取外部特征重要性文件: {IMPORTANCE_FILE} ...")

if not os.path.exists(IMPORTANCE_FILE):
    raise FileNotFoundError(f"❌ 错误：找不到文件 {IMPORTANCE_FILE}。")

try:
    df_imp = pd.read_csv(IMPORTANCE_FILE)
    col_names = df_imp.columns.str.lower()
    feat_col = next((c for c in df_imp.columns if 'feature' in c.lower()), None)
    score_col = next((c for c in df_imp.columns if 'import' in c.lower() or 'score' in c.lower()), None)

    if not feat_col or not score_col:
        if 'Feature_Name' in df_imp.columns and 'Importance_Score' in df_imp.columns:
            feat_col, score_col = 'Feature_Name', 'Importance_Score'
        elif 'Feature' in df_imp.columns and 'Importance' in df_imp.columns:
            feat_col, score_col = 'Feature', 'Importance'
        else:
            raise ValueError("无法自动识别重要性文件中的列名。")

    df_imp_sorted = df_imp.sort_values(by=score_col, ascending=False)
    top_20_features = df_imp_sorted[feat_col].head(20).tolist()

    print(f"✅ 成功加载重要性文件。")
    print(f"🏆 选定的 Top 20 特征:\n{top_20_features}")
except Exception as e:
    print(f"❌ 读取重要性文件失败: {e}")
    raise e

# ================= 2. 读取原始数据并构建特征 =================
file_path = './data/feature/csv0419_1_feature_time_sequence.csv'
print(f"\n正在读取原始数据文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 原始数据读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

df.columns = df.columns.str.strip()

if 'Label' not in df.columns:
    possible_labels = [c for c in df.columns if 'label' in c.lower() or 'type' in c.lower()]
    if possible_labels:
        df = df.rename(columns={possible_labels[0]: 'Label'})
    else:
        raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")

# 过滤罕见标签
label_counts = df['Label'].value_counts()
rare_labels = label_counts[label_counts < 2].index.tolist()
if rare_labels:
    print(f"\n⚠️  发现 {len(rare_labels)} 个罕见类别，已移除。")
    df = df[~df['Label'].isin(rare_labels)].reset_index(drop=True)

# --- 构建全量特征空间 ---
print("\n正在进行特征工程以匹配 Top 20 特征...")

feature_dict = {}

# A. 数值特征
exclude_cols = {'Label', 'excess', 'Type_Sequence', 'Type_Sequence_In_Range'}
lack_cols = [col for col in df.columns if 'lack' in col.lower()]
exclude_cols.update(lack_cols)

for col in df.columns:
    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
        feature_dict[col] = df[col].values

# B. Excess 特征
if 'excess' in df.columns:
    df['excess_parsed'] = df['excess'].apply(safe_eval_to_str_list)
    mlb_excess = MultiLabelBinarizer()
    excess_features = mlb_excess.fit_transform(df['excess_parsed'])
    excess_cols = [f"ex_{clean_feature_name(t)}" for t in mlb_excess.classes_]
    for i, name in enumerate(excess_cols):
        feature_dict[name] = excess_features[:, i]

# C. 序列特征
sequence_cols = ['Type_Sequence', 'Type_Sequence_In_Range']
for col in sequence_cols:
    if col in df.columns:
        df[f'{col}_parsed'] = df[col].apply(safe_eval_to_str_list)
        mlb_seq = MultiLabelBinarizer()
        seq_features = mlb_seq.fit_transform(df[f'{col}_parsed'])
        prefix = "seq_" if col == 'Type_Sequence' else "seq_range_"
        seq_col_names = [f"{prefix}{clean_feature_name(t)}" for t in mlb_seq.classes_]
        for i, name in enumerate(seq_col_names):
            feature_dict[name] = seq_features[:, i]

# --- 提取 Top 20 特征矩阵 ---
X_selected_list = []
final_feature_names = []
missing_features = []

for feat in top_20_features:
    if feat in feature_dict:
        X_selected_list.append(feature_dict[feat])
        final_feature_names.append(feat)
    else:
        missing_features.append(feat)

if missing_features:
    print(f"\n⚠️  警告：以下特征未匹配成功: {missing_features}")

if len(final_feature_names) == 0:
    raise ValueError("❌ 错误：没有成功匹配到任何特征。")

X_selected = np.column_stack(X_selected_list)
y = df['Label']

print(f"\n✅ 特征矩阵构建完成:")
print(f"   实际匹配特征数: {len(final_feature_names)}")
print(f"   样本数: {X_selected.shape[0]}")

# ================= 3. 数据标准化 =================
print("\n正在进行数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ================= 4. K-Means 聚类 =================
#n_clusters = len(y.unique())
n_clusters = 5
print(f"\n正在运行 K-Means 聚类 (K={n_clusters})...")
print(f"   ⚠️  注意：簇数量较多 ({n_clusters})，可能导致部分簇样本较少。")

kmeans = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42,
    algorithm='lloyd'
)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster_Label'] = cluster_labels

# ================= 5. 评估指标 =================
print(f"\n" + "=" * 50)
print(f"📊 K-Means 聚类评估结果")
print("=" * 50)

sil_score = silhouette_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)

print(f"轮廓系数 (Silhouette Score): {sil_score:.4f}")
print(f"CH Index: {ch_score:.2f}")
print(f"DB Index: {db_score:.4f}")

# ================= 6. 绘图部分 =================

# --- 图 1: 聚类簇 vs 真实标签 分布矩阵 (热力图) ---
print("\n正在绘制 [聚类簇 vs 真实标签] 分布矩阵...")

# 创建交叉表 (修复：移除 normalize=None)
cross_tab = pd.crosstab(
    df['Label'],
    df['Cluster_Label'],
    rownames=['True Label'],
    colnames=['Cluster ID']
)

# 【优化】移除全为 0 的空簇列，避免图表过宽且无意义
initial_clusters = cross_tab.columns.tolist()
non_empty_clusters = [col for col in initial_clusters if cross_tab[col].sum() > 0]
cross_tab_filtered = cross_tab[non_empty_clusters]

print(f"   原始簇数量: {len(initial_clusters)}")
print(f"   非空簇数量: {len(non_empty_clusters)} (将用于绘图)")

# 重新索引以确保顺序
unique_labels = sorted(df['Label'].unique())
cross_tab_filtered = cross_tab_filtered.reindex(index=unique_labels)

# 动态调整图形大小
width = max(14, len(non_empty_clusters) * 0.8)
height = max(8, len(unique_labels) * 0.6)

plt.figure(figsize=(width, height))
# 如果簇太多，字体调小
font_size = 10 if len(non_empty_clusters) > 30 else 12
annot_font_size = 8 if len(non_empty_clusters) > 40 else 10

sns.heatmap(
    cross_tab_filtered,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    linewidths=.5,
    cbar_kws={'label': 'Number of Samples'},
    annot_kws={"size": annot_font_size}
)

plt.title(
    f'Cluster Distribution Matrix\n(Row: True Label, Col: Cluster ID)\nTop 20 Features from RF_All_Importance.csv',
    fontsize=14)
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('True Label', fontsize=12)

# 如果簇很多，旋转 X 轴标签
if len(non_empty_clusters) > 15:
    plt.xticks(rotation=90)
else:
    plt.xticks(rotation=45)

plt.tight_layout()

save_path_matrix = os.path.join(FIG_DIR, 'Fig1_KMeans_Top20_Distribution_Matrix.png')
plt.savefig(save_path_matrix, dpi=300)
print(f"✅ 分布矩阵图已保存: {save_path_matrix}")
plt.show()

# --- 图 2: PCA 降维散点图 ---
print("\n正在生成 PCA 降维可视化...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=60, edgecolors='k',
                      linewidth=0.5)
plt.title(f'K-Means Clustering (PCA 2D)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

# 由于簇很多 (51个)，图例会非常大，这里只展示前 10 个簇的图例，或者不展示图例以免遮挡
if n_clusters > 15:
    plt.legend([], [], frameon=False)  # 隐藏图例
    plt.text(0.05, 0.95, f'Total Clusters: {n_clusters}\n(Color maps to Cluster ID)', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
else:
    handles, _ = scatter.legend_elements()
    labels_legend = [f'Cluster {i}' for i in range(n_clusters)]
    plt.legend(handles, labels_legend, title="Clusters", loc="best")

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

save_path_pca = os.path.join(FIG_DIR, 'Fig2_KMeans_Top20_PCA_Scatter.png')
plt.savefig(save_path_pca, dpi=300)
print(f"✅ PCA 散点图已保存: {save_path_pca}")
plt.show()

# --- 图 3: 使用的 Top 20 特征列表 ---
top_20_scores = []
for feat in final_feature_names:
    match = df_imp[df_imp[feat_col] == feat]
    if not match.empty:
        top_20_scores.append(match[score_col].values[0])
    else:
        top_20_scores.append(0.0)

sort_idx = np.argsort(top_20_scores)[::-1]
plot_names = [final_feature_names[i] for i in sort_idx]
plot_scores = [top_20_scores[i] for i in sort_idx]


def get_color(name):
    if name.startswith('ex_'): return '#1f77b4'
    if name.startswith('seq_'): return '#2ca02c'
    return '#ff7f0e'


colors = [get_color(name) for name in plot_names]

plt.figure(figsize=(12, 8))
plt.barh(range(len(plot_names)), plot_scores[::-1], align='center', color=colors)
plt.yticks(range(len(plot_names)), plot_names[::-1])
plt.xlabel('Importance Score (from RF_All_Importance.csv)', fontsize=12)
plt.title('Top 20 Features Used for Clustering', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path_feat = os.path.join(FIG_DIR, 'Fig3_KMeans_Top20_Features.png')
plt.savefig(save_path_feat, dpi=300)
print(f"✅ 特征列表图已保存: {save_path_feat}")
plt.show()

# ================= 7. 结果保存 =================

# 保存过滤后的交叉表 (更有意义)
output_path_cross = os.path.join(RESULT_DIR, 'KMeans_Top20_CrossTab_Filtered.csv')
cross_tab_filtered.to_csv(output_path_cross, encoding='utf-8-sig')
print(f"\n✅ 过滤后的交叉分析表已保存至: {output_path_cross}")

# 保存详细结果
result_df = df[['Label', 'Cluster_Label']].copy()
output_path_detail = os.path.join(RESULT_DIR, 'KMeans_Top20_Result.csv')
result_df.to_csv(output_path_detail, index=True, encoding='utf-8-sig', index_label='Original_Row_Index')
print(f"✅ 聚类结果详情已保存至: {output_path_detail}")

# 保存摘要
summary_file = os.path.join(RESULT_DIR, 'KMeans_Top20_Evaluation_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(f"K-Means Clustering Evaluation Summary (Top 20 External Features)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Source of Features: {IMPORTANCE_FILE}\n")
    f.write(f"Number of Clusters (K): {n_clusters}\n")
    f.write(f"Non-Empty Clusters: {len(non_empty_clusters)}\n")
    f.write(f"Features Used: {len(final_feature_names)}\n")
    f.write(f"Samples Used: {len(df)}\n\n")
    f.write("--- Metrics ---\n")
    f.write(f"Silhouette Score: {sil_score:.4f}\n")
    f.write(f"Calinski-Harabasz Score: {ch_score:.2f}\n")
    f.write(f"Davies-Bouldin Score: {db_score:.4f}\n\n")
    f.write("--- Features Used ---\n")
    for i, name in enumerate(final_feature_names):
        f.write(f"{i + 1}. {name}\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 程序执行完毕！请查看 Fig1 分布矩阵。")
print("💡 提示：由于 K=51 较大，建议观察热力图中对角线是否明显，以判断聚类是否与真实标签对应。")