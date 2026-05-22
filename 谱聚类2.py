import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    confusion_matrix,
    calinski_harabasz_score
)
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

import umap  # 需要安装 umap-learn
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数
# ==========================================
# 请根据实际情况修改路径
#IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_2_Cleaned_Final_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'
IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_3_Importance.csv'
DATA_PATH = r'.\data/feature/csv0419_1_3/D0PR_csv0419_1_3_Cleaned_Final.csv'

TOP_N_FEATURES = 12
RANDOM_STATE = 42

print(f"⚙️ 当前配置: 选取前 {TOP_N_FEATURES} 个特征")
print("🚀 正在运行优化版谱聚类分析...")

# ==========================================
# 2. 数据加载与预处理
# ==========================================
try:
    df = pd.read_csv(DATA_PATH)
    importance_df = pd.read_csv(IMPORTANCE_PATH)
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {e}")
    exit()

# 特征选择
selected_features = importance_df.head(TOP_N_FEATURES)['Feature_Name'].tolist()
X = df[selected_features]
y_true_labels = df['Label']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y_true_labels)
n_classes = len(le.classes_)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. 降维处理 (UMAP / PCA)
# ==========================================
# 谱聚类计算复杂度较高 O(N^3)，降维不仅能去噪，还能加速
print("📉 正在进行 UMAP 降维...")
reducer = umap.UMAP(n_components=min(10, X_scaled.shape[1]), random_state=RANDOM_STATE, metric='euclidean')
X_reduced = reducer.fit_transform(X_scaled)

# 如果不想用UMAP，可以用PCA：
# pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
# X_reduced = pca.fit_transform(X_scaled)

print(f"   数据维度从 {X_scaled.shape[1]} 降至 {X_reduced.shape[1]}")

# ==========================================
# 4. 改进参数搜索策略 (网格搜索 + 多指标)
# ==========================================
# 我们不仅搜索 n_neighbors，还搜索 gamma (RBF核系数)
# 使用 KNN 图构建稀疏相似度矩阵，并应用高斯核

print("🔍 正在进行网格搜索 (n_neighbors, gamma)...")

# 搜索范围
n_candidates = [5, 10, 15, 20, 30]
# gamma 候选值，通常设为 1 / (n_features * X.var()) 的附近
gamma_candidates = [0.1, 1, 10, 'scale', 'auto']

best_score = -1
best_params = {}
results_log = []

# 预计算距离矩阵或 KNN 图会更快，这里我们在循环内动态构建 affinity 矩阵
for n in n_candidates:
    if n >= len(X_reduced) - 1: continue

    # 构建 KNN 图 (包含连接性信息)
    # 注意：kneighbors_graph 返回的是二值图(0或1)，我们需要将其转换为带权重的相似度矩阵
    knn_graph = kneighbors_graph(X_reduced, n_neighbors=n, mode='connectivity', include_self=True)

    # 计算欧氏距离矩阵 (稀疏)
    # 为了计算高斯核，我们需要具体的距离值，不仅仅是连接性
    # 这里采用一种近似策略：先算距离，再mask掉非邻居的
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='auto').fit(X_reduced)
    distances, indices = nbrs.kneighbors(X_reduced)

    # 手动构建带权重的稀疏相似度矩阵 (高斯核)
    # W_ij = exp(-gamma * d^2) if j in neighbors of i else 0
    rows = np.repeat(np.arange(X_reduced.shape[0]), n)
    cols = indices.flatten()
    dists = distances.flatten()

    for gamma in gamma_candidates:
        # 处理 gamma 的特殊字符串情况
        if gamma == 'scale':
            gamma_val = 1.0 / (X_reduced.shape[1] * X_reduced.var())
        elif gamma == 'auto':
            gamma_val = 1.0 / X_reduced.shape[1]
        else:
            gamma_val = gamma

        # 计算权重
        weights = np.exp(-gamma_val * (dists ** 2))

        # 构建对称矩阵 (W + W.T) / 2 以确保图是无向的
        S = csr_matrix((weights, (rows, cols)), shape=(X_reduced.shape[0], X_reduced.shape[0]))
        S = (S + S.T) / 2

        try:
            sc = SpectralClustering(
                n_clusters=n_classes,
                affinity='precomputed',  # 使用我们自定义的相似度矩阵
                random_state=RANDOM_STATE,
                n_init=10
            )
            labels = sc.fit_predict(S.toarray())  # 注意：SpectralClustering有时对稀疏矩阵支持需转为稠密，视数据量而定

            # 只有当簇数合理时才评估
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X_reduced, labels)
                ch_score = calinski_harabasz_score(X_reduced, labels)
                # 综合评分 (简单加权，或者主要看轮廓系数)
                combined_score = sil_score  # 这里主要优化轮廓系数

                results_log.append({
                    'n_neighbors': n,
                    'gamma': gamma,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score
                })

                if combined_score > best_score:
                    best_score = combined_score
                    best_params = {'n_neighbors': n, 'gamma': gamma_val, 'S_matrix': S}
                    print(f"   👉 更新最佳: n={n}, gamma={gamma}, Silhouette={sil_score:.3f}")

        except Exception as e:
            # 谱聚类可能因为矩阵奇异等问题失败
            continue

if not best_params:
    print("❌ 未能找到合适的参数组合，请检查数据或扩大搜索范围。")
    exit()

print("-" * 30)
print(f"✅ 最佳参数: n_neighbors={best_params['n_neighbors']}, gamma={best_params['gamma']}")
print("-" * 30)

# ==========================================
# 5. 执行最终聚类
# ==========================================
final_clustering = SpectralClustering(
    n_clusters=n_classes,
    affinity='precomputed',
    random_state=RANDOM_STATE,
    n_init=20  # 增加初始化次数以提高稳定性
)
y_pred = final_clustering.fit_predict(best_params['S_matrix'].toarray())

# 计算指标
ari = adjusted_rand_score(y_encoded, y_pred)
final_silhouette = silhouette_score(X_reduced, y_pred)
print(f"📊 最终结果: ARI={ari:.3f}, Silhouette={final_silhouette:.3f}")


# ==========================================
# 6. 解决标签不对齐 (匈牙利算法)
# ==========================================
# 谱聚类的标签是 0, 1, 2... 但它们是随机分配的，不一定对应真实标签的 0, 1, 2
# 我们使用匈牙利算法找到最佳的一一映射

def align_labels(true_labels, pred_labels):
    """
    使用匈牙利算法将预测标签映射到真实标签空间，以最大化匹配度
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    # 匈牙利算法求解的是最小化成本，所以我们取负数
    # linear_sum_assignment 返回的是 (行索引, 列索引)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # 创建一个映射数组
    # map[i] = j 表示 预测簇 j 应该被重命名为 真实标签 i
    # 但我们需要的是：新标签[pred_label] = true_label
    label_mapping = np.zeros(len(col_ind), dtype=int)
    for true_idx, pred_idx in zip(row_ind, col_ind):
        label_mapping[pred_idx] = true_idx

    # 应用映射
    aligned_pred_labels = np.array([label_mapping[p] for p in pred_labels])
    return aligned_pred_labels, cm, label_mapping


y_pred_aligned, original_cm, mapping = align_labels(y_encoded, y_pred)

# ==========================================
# 7. 绘制混淆矩阵 (对齐后)
# ==========================================
plt.figure(figsize=(12, 10))

# 注意：这里使用对齐后的 y_pred_aligned 来计算混淆矩阵，对角线应该很完美
# 但为了展示“修正前”和“修正后”的区别，我们可以画修正后的
cm_display = confusion_matrix(y_encoded, y_pred_aligned)

sns.heatmap(
    cm_display,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,  # 现在 X 轴也是真实标签名了，因为已经对齐
    yticklabels=le.classes_,
    cbar_kws={'label': '样本数量'}
)

plt.title(
    f'谱聚类结果 (标签对齐后)\n(ARI: {ari:.3f} | Silhouette: {final_silhouette:.3f})',
    fontsize=15
)
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('聚类标签 (已自动对齐)', fontsize=12)
plt.tight_layout()

output_path = "Spectral_Clustering_Aligned_Matrix.png"
plt.savefig(output_path, dpi=300)
print(f"🖼️ 混淆矩阵已保存为: {output_path}")
plt.show()

# 打印标签映射关系供参考
print("🗺️ 标签映射关系 (原始预测簇 -> 真实标签):")
for i, true_label_idx in enumerate(mapping):
    print(f"   Cluster {i} -> {le.inverse_transform([true_label_idx])[0]}")