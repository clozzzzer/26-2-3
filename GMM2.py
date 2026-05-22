import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

# 机器学习库
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture

# 新增：UMAP 库
import umap

import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数
# ==========================================
IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_2_Cleaned_Final_Importance.csv'
DATA_PATH = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'

RANDOM_SEED = 42
SELECTED_N_CLASSES = 25  # 随机抽取的类别数量
TOP_N_FEATURES = 12  # 🚀 现在可以选更多特征了！UMAP 会处理可视化 (之前受限于2)

print(f"🚀 正在加载数据 (目标类别数: {SELECTED_N_CLASSES}, 特征数: {TOP_N_FEATURES})...")

# ==========================================
# 2. 数据加载与随机类别选择
# ==========================================
try:
    df = pd.read_csv(DATA_PATH)
    df_imp = pd.read_csv(IMPORTANCE_PATH)
except FileNotFoundError as e:
    print(f"❌ 文件错误: {e}")
    exit()

# 随机选择类别
all_labels = df['Label'].unique()
if SELECTED_N_CLASSES >= len(all_labels):
    selected_labels = all_labels
else:
    random.seed(RANDOM_SEED)
    selected_labels = random.sample(list(all_labels), SELECTED_N_CLASSES)

print(f"🎲 选中类别 ({len(selected_labels)}): {sorted(selected_labels)}")
df_filtered = df[df['Label'].isin(selected_labels)].copy()

# 标签编码
le = LabelEncoder()
df_filtered['Label_Encoded'] = le.fit_transform(df_filtered['Label'])
y_true = df_filtered['Label_Encoded']
true_names = df_filtered['Label']

# 特征选择 (现在可以选 Top 10 或更多)
imp_col = [c for c in df_imp.columns if 'importance' in c.lower() or 'score' in c.lower()][0]
top_features = df_imp.sort_values(by=imp_col, ascending=False).head(TOP_N_FEATURES)['Feature_Name'].tolist()
X = df_filtered[top_features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. UMAP 降维 (用于可视化)
# ==========================================
print("\n📉 正在进行 UMAP 降维 (用于绘图)...")
reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, metric='euclidean')
X_umap = reducer.fit_transform(X_scaled)
print("✅ UMAP 降维完成")

# ==========================================
# 4. GMM 参数自动寻优 (BIC准则)
# ==========================================
print("\n🔍 正在寻找最佳协方差类型...")
covariance_types = ['full', 'tied', 'diag', 'spherical']
best_bic = np.inf
best_model = None
best_cov_type = ''

for cov_type in covariance_types:
    try:
        gmm = GaussianMixture(
            n_components=len(selected_labels),
            covariance_type=cov_type,
            random_state=RANDOM_SEED,
            reg_covar=1e-4,
            max_iter=200
        )
        gmm.fit(X_scaled)  # 注意：GMM 依然在原始高维数据上训练
        bic = gmm.bic(X_scaled)
        print(f" - {cov_type:6s}: BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_cov_type = cov_type
    except Exception as e:
        print(f" - {cov_type:6s}: 失败 ({e})")

print(f"✅ 最佳模型: {best_cov_type} (BIC: {best_bic:.2f})")

# ==========================================
# 5. 训练最佳模型 & 预测
# ==========================================
probs = best_model.predict_proba(X_scaled)
y_pred_hard = best_model.predict(X_scaled)
max_probs = np.max(probs, axis=1)

ari = adjusted_rand_score(y_true, y_pred_hard)
silhouette = silhouette_score(X_scaled, y_pred_hard)
avg_confidence = np.mean(max_probs)

print(f"\n📈 性能指标: ARI={ari:.3f}, 轮廓系数={silhouette:.3f}, 平均置信度={avg_confidence:.3f}")


# ==========================================
# 6. 标签对齐
# ==========================================
def align_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    y_pred_aligned = np.array([mapping.get(p, -1) for p in y_pred])
    return y_pred_aligned, mapping


y_pred_aligned, label_map = align_labels(y_true, y_pred_hard)
print(f"🔗 标签映射关系 (预测簇 -> 真实标签): {label_map}")

# ==========================================
# 7. 绘图：混淆矩阵 (独立窗口)
# ==========================================
print("\n🎨 正在生成独立混淆矩阵窗口...")
plt.figure(figsize=(10, 8))
cm_aligned = confusion_matrix(y_true, y_pred_aligned)
sns.heatmap(
    cm_aligned,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title(f'✅ 聚类混淆矩阵 (对齐后)\nAdjusted Rand Index: {ari:.3f}', fontsize=16)
plt.xlabel('预测标签 (Predicted)', fontsize=12)
plt.ylabel('真实标签 (True Label)', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# 8. 绘图：UMAP 可视化 + GMM 椭圆 (主窗口)
# ==========================================
print("📊 正在生成 UMAP 可视化窗口...")
fig = plt.figure(figsize=(16, 6))

# --- 子图 1: UMAP 散点图 ---
ax1 = plt.subplot(1, 2, 1)
scatter = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=y_pred_aligned, cmap='tab20', s=30, alpha=0.6, label='Samples')

# 🚀 在 UMAP 空间绘制椭圆
# 注意：GMM 的均值和协方差是在原始高维空间的。
# 为了在 2D 图上画椭圆，我们需要将高维的均值/协方差“投影”到 UMAP 空间。
# 方法：对 GMM 的均值进行 transform，对协方差进行近似投影。

colors = plt.cm.tab20(np.linspace(0, 1, len(selected_labels)))

# 将高维均值投影到 2D
means_2d = reducer.transform(best_model.means_)

for i in range(len(selected_labels)):
    mean_2d = means_2d[i]
    color = colors[i % len(colors)]

    # 协方差投影 (近似)
    # 获取原始协方差矩阵
    if best_cov_type == 'tied':
        cov_orig = best_model.covariances_
    elif best_cov_type == 'diag':
        cov_orig = np.diag(best_model.covariances_[i])
    elif best_cov_type == 'spherical':
        cov_orig = np.eye(X_scaled.shape[1]) * best_model.covariances_[i]
    else:  # full
        cov_orig = best_model.covariances_[i]

    # 近似投影协方差到 2D: J * Cov * J.T
    # 由于 UMAP 是非线性的，没有精确的 Jacobian。
    # 这里使用一种简化的启发式方法：计算投影后均值的局部协方差，
    # 或者直接根据投影后的数据估算该簇的 2D 协方差。

    # 简单策略：找出属于该簇的点，计算它们在 UMAP 空间里的协方差
    cluster_points_umap = X_umap[y_pred_hard == i]
    if len(cluster_points_umap) > 1:
        cov_2d = np.cov(cluster_points_umap, rowvar=False)

        # 绘制椭圆
        from matplotlib.patches import Ellipse

        vals, vecs = np.linalg.eigh(cov_2d)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # 2个标准差
        width, height = 2 * np.sqrt(vals) * 2
        ell = Ellipse(xy=mean_2d, width=width, height=height, angle=theta,
                      color=color, fill=False, linestyle='--', alpha=0.8, linewidth=2)
        ax1.add_artist(ell)

ax1.set_title(f'UMAP 可视化 (基于 Top {TOP_N_FEATURES} 特征)', fontsize=14)
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')

# --- 子图 2: 置信度分布 ---
ax2 = plt.subplot(1, 2, 2)
correct_mask = (y_true == y_pred_aligned)
sns.kdeplot(max_probs[correct_mask], fill=True, color='green', label='正确分类 (Correct)', alpha=0.6)
if np.sum(~correct_mask) > 0:
    sns.kdeplot(max_probs[~correct_mask], fill=True, color='red', label='错误分类 (Incorrect)', alpha=0.6)

ax2.set_title('分类置信度 (Posterior Probability)', fontsize=14)
ax2.set_xlabel('最大概率值')
ax2.set_ylabel('密度')
ax2.legend()

plt.suptitle(f'GMM + UMAP 聚类分析报告', fontsize=16)
plt.tight_layout()
plt.show()