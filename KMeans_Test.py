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
import random

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

FIG_DIR = './fig'
RESULT_DIR = './Result'
TARGET_CLASS_COUNT = 15  # ⬅️ 目标随机抽取的类别数量
RANDOM_SEED = None  # 设置随机种子以保证结果可复现，如需每次不同可设为 None

if RANDOM_SEED:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

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


# ================= 1. 读取数据 =================
# 请根据实际文件路径修改此处，原代码可能是 csv0419_1_feature_time_sequence.csv
file_path = './data/feature/csv0419_1/All_feature_csv0419_1.csv'
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
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

print(f"原始数据形状: {df.shape}")
print(f"原始类别总数: {df['Label'].nunique()}")

# ================= 2. 🎯 核心修改：随机抽取 15 个类 =================
print(f"\n🔄 正在执行类别采样策略 (目标: {TARGET_CLASS_COUNT} 类)...")

# A. 统计样本分布
label_counts = df['Label'].value_counts()

# B. 过滤掉样本数 < 2 的类别 (KMeans 需要至少 2 个样本才能形成有效的簇，且避免划分训练测试集时报错)
valid_labels = label_counts[label_counts >= 2].index.tolist()
removed_rare = label_counts[label_counts < 2].index.tolist()

if removed_rare:
    print(f"⚠️  已移除 {len(removed_rare)} 个样本数不足 (<2) 的罕见类别。")

if len(valid_labels) < TARGET_CLASS_COUNT:
    print(f"❌ 错误：有效类别数量 ({len(valid_labels)}) 少于目标数量 ({TARGET_CLASS_COUNT})。")
    print(f"   有效类别列表: {valid_labels}")
    raise ValueError("无法抽取足够的类别。请检查数据或减少 TARGET_CLASS_COUNT。")

# C. 随机抽取
selected_labels = random.sample(valid_labels, TARGET_CLASS_COUNT)
selected_labels.sort()  # 排序以便展示

print(f"✅ 成功随机抽取 {TARGET_CLASS_COUNT} 个类别:")
for lbl in selected_labels:
    count = label_counts[lbl]
    print(f"   - {lbl} (样本数: {count})")

# D. 过滤数据集
df_filtered = df[df['Label'].isin(selected_labels)].reset_index(drop=True)
y = df_filtered['Label']

print(f"\n📉 数据子集构建完成:")
print(f"   原始样本数: {len(df)} -> 子集样本数: {len(df_filtered)}")
print(f"   原始类别数: {df['Label'].nunique()} -> 子集类别数: {y.nunique()}")

# 更新 df 为过滤后的数据，后续步骤均基于此子集
df = df_filtered

# ================= 3. 特征工程 =================
print("\n正在进行特征工程...")

feature_parts = []
feature_names_total = []

# --- A. 处理 Excess 特征 (如果存在) ---
if 'excess' in df.columns:
    df['excess_parsed'] = df['excess'].apply(safe_eval_to_str_list)
    mlb_excess = MultiLabelBinarizer()
    excess_features = mlb_excess.fit_transform(df['excess_parsed'])
    excess_cols = [f"ex_{clean_feature_name(t)}" for t in mlb_excess.classes_]
    feature_parts.append(excess_features)
    feature_names_total.extend(excess_cols)

# --- B. 处理序列特征 (核心部分) ---
sequence_cols = ['Type_Sequence', 'Type_Sequence_In_Range']
found_seq_cols = [col for col in sequence_cols if col in df.columns]

if found_seq_cols:
    for col in found_seq_cols:
        df[f'{col}_parsed'] = df[col].apply(safe_eval_to_str_list)
        mlb_seq = MultiLabelBinarizer()
        seq_features = mlb_seq.fit_transform(df[f'{col}_parsed'])
        prefix = "seq_" if col == 'Type_Sequence' else "seq_range_"
        seq_col_names = [f"{prefix}{clean_feature_name(t)}" for t in mlb_seq.classes_]
        feature_parts.append(seq_features)
        feature_names_total.extend(seq_col_names)

# --- C. 提取其他数值特征 ---
exclude_cols = {
    'Label', 'excess', 'excess_parsed',
    'Type_Sequence', 'Type_Sequence_parsed',
    'Type_Sequence_In_Range', 'Type_Sequence_In_Range_parsed'
}
# 排除 lack 列 (根据原代码逻辑)
lack_cols = [col for col in df.columns if 'lack' in col.lower()]
exclude_cols.update(lack_cols)

numeric_candidate_cols = [col for col in df.columns if col not in exclude_cols]
valid_numeric_cols = [col for col in numeric_candidate_cols if pd.api.types.is_numeric_dtype(df[col])]

if valid_numeric_cols:
    numeric_data = df[valid_numeric_cols].values
    # 处理可能的 NaN
    if np.isnan(numeric_data).any():
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)
    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)

# --- D. 合并所有特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X_full = np.hstack(feature_parts)
print(f"✅ 原始特征矩阵构建完成: {X_full.shape[1]} 个特征, 样本数: {X_full.shape[0]}")

# ================= 4. 数据标准化 =================
print("\n正在进行数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# ================= 5. K-Means 聚类 =================
# 关键修改：聚类簇数直接设置为选定的类别数 (15)
n_clusters = TARGET_CLASS_COUNT
print(f"\n正在运行 K-Means 聚类 (K={n_clusters})...")

kmeans = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=RANDOM_SEED,
    algorithm='lloyd'
)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster_Label'] = cluster_labels

# ================= 6. 评估指标 =================
print(f"\n" + "=" * 50)
print(f"📊 K-Means 聚类评估结果 (随机 {TARGET_CLASS_COUNT} 类)")
print("=" * 50)

sil_score = silhouette_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)

print(f"轮廓系数 (Silhouette Score): {sil_score:.4f}")
print(f"CH Index: {ch_score:.2f}")
print(f"DB Index: {db_score:.4f}")

# ================= 7. 绘图部分 =================

# --- 图 1: 聚类簇 vs 真实标签 分布矩阵 (热力图) ---
print("\n正在绘制 [聚类簇 vs 真实标签] 分布矩阵...")

cross_tab = pd.crosstab(
    df['Label'],
    df['Cluster_Label'],
    rownames=['True Label'],
    colnames=['Cluster ID']
)

# 确保行列顺序一致且美观
unique_labels = sorted(df['Label'].unique())
unique_clusters = sorted(df['Cluster_Label'].unique())
cross_tab = cross_tab.reindex(index=unique_labels, columns=unique_clusters)

plt.figure(figsize=(12, 10))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Number of Samples'})
plt.title(
    f'Cluster Distribution Matrix\n(Row: True Label, Col: Cluster ID)\nRandomly Selected {TARGET_CLASS_COUNT} Classes',
    fontsize=14)
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

save_path_matrix = os.path.join(FIG_DIR, f'Fig1_KMeans_Random{TARGET_CLASS_COUNT}_Matrix.png')
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

# 添加图例
handles, _ = scatter.legend_elements()
labels_legend = [f'Cluster {i}' for i in range(n_clusters)]
plt.legend(handles, labels_legend, title="Clusters", loc="best", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

save_path_pca = os.path.join(FIG_DIR, f'Fig2_KMeans_Random{TARGET_CLASS_COUNT}_PCA.png')
plt.savefig(save_path_pca, dpi=300)
print(f"✅ PCA 散点图已保存: {save_path_pca}")
plt.show()

# ================= 8. 结果保存 =================

# 保存交叉表
output_path_cross = os.path.join(RESULT_DIR, f'KMeans_Random{TARGET_CLASS_COUNT}_CrossTab.csv')
cross_tab.to_csv(output_path_cross, encoding='utf-8-sig')
print(f"\n✅ 交叉分析表已保存至: {output_path_cross}")

# 保存详细结果
result_df = df[['Label', 'Cluster_Label']].copy()
output_path_detail = os.path.join(RESULT_DIR, f'KMeans_Random{TARGET_CLASS_COUNT}_Result.csv')
result_df.to_csv(output_path_detail, index=True, encoding='utf-8-sig', index_label='Original_Row_Index')
print(f"✅ 聚类结果详情已保存至: {output_path_detail}")

# 保存摘要
summary_file = os.path.join(RESULT_DIR, f'KMeans_Random{TARGET_CLASS_COUNT}_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(f"K-Means Clustering Summary (Random {TARGET_CLASS_COUNT} Classes)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Random Seed: {RANDOM_SEED}\n")
    f.write(f"Selected Classes: {selected_labels}\n")
    f.write(f"Total Samples Used: {len(df)}\n")
    f.write(f"Number of Clusters (K): {n_clusters}\n\n")
    f.write("--- Metrics ---\n")
    f.write(f"Silhouette Score: {sil_score:.4f}\n")
    f.write(f"Calinski-Harabasz Score: {ch_score:.2f}\n")
    f.write(f"Davies-Bouldin Score: {db_score:.4f}\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 程序执行完毕！")
print(f"💡 提示：本次运行随机选择了 {selected_labels} 进行聚类。")