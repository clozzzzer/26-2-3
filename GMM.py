import os
import random  # <--- 新增：用于随机抽样
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
# 设置绘图中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数
# ==========================================
#IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_Cleaned_Final_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1/D0PR_csv0419_1_Cleaned_Final.csv'
# ==========================================
#IMPORTANCE_PATH = r'.\Result/RF_All/RF_D1PR_csv0419_1_Cleaned_Final_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1/D1PR_csv0419_1_Cleaned_Final.csv'
# ==========================================
IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0PR_csv0419_1_2_Cleaned_Final_Importance.csv'
DATA_PATH = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'

# 🎯 核心修改参数
RANDOM_SEED = 42             # 随机种子，保证结果可复现
SELECTED_N_CLASSES = 25    # <--- 在这里修改：想随机选多少个类别？(例如 5, 10, 15)
TOP_N_FEATURES = 2           # 选取前 N 个重要特征
COVARIANCE_TYPE = 'diag'      # GMM 协方差类型 ('full', 'tied', 'diag', 'spherical')

# 设置随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"🚀 正在加载数据 (随机种子: {RANDOM_SEED}, 目标类别数: {SELECTED_N_CLASSES})...")

# ==========================================
# 2. 数据加载与随机类别选择
# ==========================================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {DATA_PATH}")
    exit()

# --- 随机选择类别逻辑开始 ---
all_labels = df['Label'].unique()
total_available_classes = len(all_labels)

if SELECTED_N_CLASSES >= total_available_classes:
    print(f"⚠️ 提示: 目标类别数 ({SELECTED_N_CLASSES}) 大于等于总类别数 ({total_available_classes})，将使用所有类别。")
    selected_labels = all_labels
else:
    # 随机抽取 n 个类别
    selected_labels = random.sample(list(all_labels), SELECTED_N_CLASSES)

print(f"🎲 从 {total_available_classes} 个总类别中随机选中了 {len(selected_labels)} 个类别:")
print(f"   {sorted(selected_labels)}")

# 过滤数据，只保留选中的类别
df_filtered = df[df['Label'].isin(selected_labels)].copy()

# 重新编码标签 (将选中的标签映射为 0 到 N-1，这对聚类和绘图至关重要)
le = LabelEncoder()
df_filtered['Label_Encoded'] = le.fit_transform(df_filtered['Label'])

print(f"📊 过滤后的数据形状: {df_filtered.shape}")
# --- 随机选择类别逻辑结束 ---

# ==========================================
# 3. 特征选择
# ==========================================
try:
    df_imp = pd.read_csv(IMPORTANCE_PATH)
    # 自动检测重要性列名 (兼容 'Importance', 'importance', 'Score' 等)
    imp_col = [c for c in df_imp.columns if 'importance' in c.lower() or 'score' in c.lower() or 'value' in c.lower()]
    if not imp_col:
        raise ValueError("在 importance.csv 中未找到重要性列")
    imp_col = imp_col[0]

    # 获取前 N 个特征名
    top_features = df_imp.sort_values(by=imp_col, ascending=False).head(TOP_N_FEATURES)['Feature_Name'].tolist()

    # 确保这些特征都在主数据集中
    available_features = [f for f in top_features if f in df_filtered.columns]

    if len(available_features) < TOP_N_FEATURES:
        print(f"⚠️ 警告: 只有 {len(available_features)} 个特征可用，少于请求的 {TOP_N_FEATURES}")

    X = df_filtered[available_features]
    y_true = df_filtered['Label_Encoded']
    true_labels_names = df_filtered['Label'] # 用于绘图时的标签显示

except Exception as e:
    print(f"❌ 特征选择出错: {e}")
    exit()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"⚙️ 当前配置: 选取前 {len(available_features)} 个特征 (模式: {COVARIANCE_TYPE})")

# ==========================================
# 4. 训练 GMM 模型
# ==========================================
print(f"\n⚙️ 正在训练 GMM 模型 (n_components={len(selected_labels)}, covariance_type='{COVARIANCE_TYPE}')...")

# 注意：n_components 应该等于我们选中的类别数
gmm = GaussianMixture(
    n_components=len(selected_labels),
    covariance_type=COVARIANCE_TYPE,
    random_state=RANDOM_SEED,
    reg_covar=1e-4  # 增加正则化防止奇异矩阵
)
gmm.fit(X_scaled)
y_pred = gmm.predict(X_scaled)

# ==========================================
# 5. 评估与可视化
# ==========================================
ari = adjusted_rand_score(y_true, y_pred)
silhouette = silhouette_score(X_scaled, y_pred)

print(f"\n📈 聚类性能指标:")
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")

# 簇成分分析
print("\n🔎 簇成分分析 (每个簇中数量最多的真实标签):")
results = []
for i in range(len(selected_labels)):
    cluster_mask = (y_pred == i)
    if np.sum(cluster_mask) == 0:
        continue

    # 获取该簇对应的真实标签名称
    true_labels_in_cluster = true_labels_names[cluster_mask]
    dominant_label = true_labels_in_cluster.mode()[0]
    purity = true_labels_in_cluster.value_counts().max() / np.sum(cluster_mask)
    count = np.sum(cluster_mask)

    results.append({
        "Cluster": i,
        "Dominant_Label": dominant_label,
        "Purity": purity,
        "Count": count
    })

results_df = pd.DataFrame(results)
# 按簇索引排序显示
results_df = results_df.sort_values(by="Cluster")
print(results_df.to_string(index=False))

# ==========================================
# 6. 绘制混淆矩阵
# ==========================================
plt.figure(figsize=(16, 14))

# 使用重新编码后的标签计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 获取选中的标签名称用于坐标轴
label_names = le.classes_

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[f'Cluster {i}' for i in range(len(selected_labels))],
    yticklabels=label_names,
    cbar_kws={'label': '样本数量'}
)

plt.title(f'Confusion Matrix (Selected {len(selected_labels)} Classes)\nARI: {ari:.3f} | Silhouette: {silhouette:.3f}', fontsize=16)
plt.xlabel('Predicted Cluster', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()