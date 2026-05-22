import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
import warnings
import os
from scipy.optimize import linear_sum_assignment  # 确保导入在这里

# 忽略警告
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数
# ==========================================
#IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_2_Cleaned_Final_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'
IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_3_Importance.csv'
DATA_PATH = r'.\data/feature/csv0419_1_3/D0PR_csv0419_1_3_Cleaned_Final.csv'

TOP_N_FEATURES = 10
RANDOM_STATE = 42

print(f"⚙️ 当前配置: 选取前 {TOP_N_FEATURES} 个特征")
print("🚀 正在运行谱聚类分析...")

# ==========================================
# 2. 数据加载与预处理
# ==========================================
# 加载数据
df = pd.read_csv(DATA_PATH)
importance_df = pd.read_csv(IMPORTANCE_PATH)

# 筛选前 N 个重要特征
selected_features = importance_df.head(TOP_N_FEATURES)['Feature_Name'].tolist()

# 提取特征和标签
X = df[selected_features]
y_true_labels = df['Label']  # 假设 'Label' 是你的真实类别列名

# 标签编码 (将字符串标签转为数字，用于计算指标)
le = LabelEncoder()
y_encoded = le.fit_transform(y_true_labels)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. 寻找最佳 n_neighbors (谱聚类的关键参数)
# ==========================================
# 谱聚类对 n_neighbors 很敏感，我们尝试几个值看哪个轮廓系数最高
best_score = -1
best_n = 5
n_candidates = [5, 10, 15, 20, 30]
print("🔍 正在寻找最佳 n_neighbors...")

for n in n_candidates:
    # 确保 n_neighbors 小于样本数的一半，防止全连通
    if n >= len(X_scaled) // 2:
        continue
    sc = SpectralClustering(n_clusters=len(le.classes_),
                            affinity='nearest_neighbors',
                            n_neighbors=n,
                            random_state=RANDOM_STATE,
                            n_init=10)
    labels = sc.fit_predict(X_scaled)

    # 计算轮廓系数 (只有当簇数 > 1 且 < 样本数时才有意义)
    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
        print(f" - n_neighbors={n}: 轮廓系数={score:.3f}")
        if score > best_score:
            best_score = score
            best_n = n

print(f"✅ 最佳 n_neighbors: {best_n} (轮廓系数: {best_score:.3f})")

# ==========================================
# 4. 执行最终聚类
# ==========================================
final_clustering = SpectralClustering(
    n_clusters=len(le.classes_),
    affinity='nearest_neighbors',
    n_neighbors=best_n,
    random_state=RANDOM_STATE,
    n_init=10
)
y_pred = final_clustering.fit_predict(X_scaled)

# 计算评估指标
ari = adjusted_rand_score(y_encoded, y_pred)
n_clusters = len(set(y_pred))
print("-" * 30)
print(f"📊 最终结果: N_Clusters={n_clusters}, ARI={ari:.3f}")
print("-" * 30)

# ==========================================
# 5. 绘制优化版混淆矩阵 (解决标签对齐问题)
# ==========================================
print("🎨 正在生成优化后的混淆矩阵...")


# --- 步骤 1: 解决标签不对齐 (核心逻辑) ---
def find_best_mapping(y_true, y_pred):
    # 获取唯一标签
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    n_true = len(unique_true)
    n_pred = len(unique_pred)

    # 创建混淆矩阵作为成本矩阵 (行: 真实, 列: 预测)
    # 我们的目标是找到预测标签到真实标签的最佳排列
    cost_matrix = np.zeros((n_pred, n_true))

    for i, pred_label in enumerate(unique_pred):
        for j, true_label in enumerate(unique_true):
            # 计算将 pred_label 映射为 true_label 时的"错误"数量
            # 这里我们直接用混淆矩阵的逻辑
            mask_pred = (y_pred == pred_label)
            mask_true = (y_true == true_label)
            # 错误数 = 预测为 pred_label 但真实不是 true_label 的数量
            # 为了最大化正确率，我们最小化错误率
            incorrect = np.sum(mask_pred) - np.sum(mask_pred & mask_true)
            cost_matrix[i, j] = incorrect

    # 使用匈牙利算法求解最小成本
    # row_ind 是预测标签的索引, col_ind 是真实标签的索引
    # 这意味着: 预测标签 row_ind[i] 应该被映射为 真实标签 col_ind[i]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 构建映射字典: pred_label -> true_label
    mapping = {}
    for pred_idx, true_idx in zip(row_ind, col_ind):
        mapping[unique_pred[pred_idx]] = unique_true[true_idx]

    # 生成对齐后的预测标签
    y_pred_aligned = np.array([mapping.get(label, -1) for label in y_pred])
    return y_pred_aligned, mapping


# 计算对齐后的标签
try:
    y_pred_final, label_map = find_best_mapping(y_encoded, y_pred)
    print("✅ 标签已自动对齐。映射关系 (预测簇 -> 真实标签索引):", label_map)
except Exception as e:
    print(f"⚠️ 标签对齐失败: {e}，使用原始预测标签。")
    y_pred_final = y_pred

# --- 步骤 2: 计算混淆矩阵 ---
cm = confusion_matrix(y_encoded, y_pred_final)

# --- 步骤 3: 绘图设置 ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 1. 绝对数量热力图
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[0],
            cbar_kws={'label': '样本数量'})
axes[0].set_title(f'1. 混淆矩阵 (样本数量)\n(对角线为正确分类)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('真实标签 (True Label)', fontsize=12)
axes[0].set_xlabel('预测标签 (Predicted Label)', fontsize=12)

# 2. 归一化比例热力图 (显示每个类别的准确率)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[1],
            cbar_kws={'label': '比例'})
axes[1].set_title(f'2. 混淆矩阵 (归一化比例)\n(对角线为分类准确率)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('真实标签 (True Label)', fontsize=12)
axes[1].set_xlabel('预测标签 (Predicted Label)', fontsize=12)

# 总标题
plt.suptitle(f'谱聚类结果可视化 | ARI: {ari:.3f} | 轮廓系数: {best_score:.3f}',
             fontsize=16, y=1.02)

plt.tight_layout()
output_path = "Spectral_Clustering_Confusion_Matrix_Enhanced.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"🖼️ 增强版混淆矩阵已保存为: {output_path}")
plt.show()

# --- 步骤 4: 打印详细指标 ---
print("\n📋 详细分类报告:")
print("-" * 50)
for i, class_name in enumerate(le.classes_):
    tp = cm[i, i]  # 真正例
    total_true = cm[i, :].sum()  # 真实总数
    accuracy = (tp / total_true) * 100 if total_true > 0 else 0
    print(f"{class_name:12s} : 准确率 {accuracy:6.2f}% ( {tp:3d} / {total_true:3d} )")