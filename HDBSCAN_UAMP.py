import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# --- 新增导入：UMAP ---
import umap

# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数 (新增 UMAP 相关参数)
# ==========================================
#IMPORTANCE_PATH = r'./Result/RF_All\RF_All_Cleaned_feature_csv0419_1_2_Importance.csv'
#DATA_PATH = 'data/feature/csv0419_1_2/D0PR_All_Cleaned_feature_csv0419_1_2.csv'
#IMPORTANCE_PATH = r'.\Result/RF_All/RF_D0NOPR_csv0419_2_Cleaned_Final_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1/D0NOPR_csv0419_2_Cleaned_Final.csv'
#IMPORTANCE_PATH = r'.\Result/RF_All/RF_D1PR_csv0419_1_Cleaned_Final_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1/D1PR_csv0419_1_Cleaned_Final.csv'
#IMPORTANCE_PATH = r'./Result/RF_All\RF_All_Cleaned_feature_csv0419_1_4_Importance.csv'
IMPORTANCE_PATH = r'Result/Funnel_Strategy_Extended/Final_Feature_Importance.csv'
DATA_PATH = r'.\data/feature/csv0419_1_4/D0PR_All_Cleaned_feature_csv0419_1_4.csv'
#IMPORTANCE_PATH = r'Result/Funnel_Strategy_Extended/D0PR_All_Cleaned_feature_csv0419_1_5_2/Final_Feature_Importance.csv'
#DATA_PATH = r'data/feature/csv0419_1_5_2/D0PR_All_Cleaned_feature_csv0419_1_5_2.csv'

TOP_N_FEATURES = 9  # 选择前 N 个重要特征
RANDOM_STATE = 42

# --- HDBSCAN 参数调整 ---
# 原 min_cluster_size 在高维稀疏数据中可能需要调大，但在 UMAP 降维后，簇会更紧密，可以适当调小或保持

# UMAP 降维目标维度 (通常 2-10 维效果很好，既能降噪又能保留结构)
N_COMPONENTS_UMAP = 2
n_nei = 5
min_d = 0.025
MIN_CLUSTER_SIZE = 4
print(
    f"⚙️ 当前配置: 选取前 {TOP_N_FEATURES} 个特征, HDBSCAN min_cluster_size={MIN_CLUSTER_SIZE}, UMAP dimensions={N_COMPONENTS_UMAP}")


# ==========================================
# 2. 数据加载与预处理 (包含随机抽取15类逻辑)
# ==========================================
def load_and_sample_data(data_path, n_classes=16, random_state=42):
    print("1. 📂 正在加载数据并随机抽取类别...")
    # 读取数据
    df = pd.read_csv(data_path)

    # 检查是否有标签列 (假设标签列名为 'Label' 或 'label')
    label_col = 'Label' if 'Label' in df.columns else 'label'
    if label_col not in df.columns:
        raise ValueError(f"未找到标签列 'Label' 或 'label'，请检查列名。当前列名: {df.columns.tolist()}")

    # 获取所有唯一的标签
    all_labels = df[label_col].unique()
    print(f" - 原始数据集包含 {len(all_labels)} 个类别")

    # 随机抽取 n_classes 个类别
    np.random.seed(random_state)
    selected_labels = np.random.choice(all_labels, n_classes, replace=False)
    print(f" - 🎲 随机抽取的 {n_classes} 个类别: {selected_labels}")

    # 筛选数据
    df_filtered = df[df[label_col].isin(selected_labels)].copy()

    # 分离特征和标签
    y_true = df_filtered[label_col].values  # 返回 numpy 数组
    X_data = df_filtered.drop(columns=[label_col])

    print(f" - ✅ 筛选后数据量: {X_data.shape[0]} 行, {X_data.shape[1]} 个特征")
    return X_data, y_true, selected_labels


# ==========================================
# 3. 特征加权、UMAP降维与标准化
# ==========================================
def prepare_features(X_data, importance_path, top_n):
    print("2. ⚙️ 正在根据特征重要性进行加权与UMAP降维...")

    # --- 3.1 特征选择与加权 (同原逻辑) ---
    imp_df = pd.read_csv(importance_path)
    if 'Feature_Name' not in imp_df.columns:
        imp_df.columns = ['Feature_Name', 'Importance_Score']  # 简单重命名

    imp_df_sorted = imp_df.sort_values(by='Importance_Score', ascending=False).head(top_n)
    selected_features = imp_df_sorted['Feature_Name'].tolist()

    # 检查数据中是否包含这些特征
    missing_cols = [c for c in selected_features if c not in X_data.columns]
    if missing_cols:
        print(f" ⚠️ 警告: 数据中缺少 {len(missing_cols)} 个重要性特征，已自动跳过。")
        selected_features = [c for c in selected_features if c in X_data.columns]

    X_selected = X_data[selected_features]
    weights = imp_df_sorted.set_index('Feature_Name').loc[selected_features, 'Importance_Score'].values

    # 标准化 (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 加权 (利用 numpy 的广播机制)
    X_weighted = X_scaled * weights
    print(f" - ✅ 加权完成，维度: {X_weighted.shape}")

    # --- 3.2 UMAP 降维 ---
    # 注意: UMAP 本身对距离敏感，我们在这里先用欧氏距离构建图，但在低维空间中它会形成流形
    print(f" - 📉 正在使用 UMAP 降维至 {N_COMPONENTS_UMAP} 维...")
    reducer = umap.UMAP(
        n_components=N_COMPONENTS_UMAP,
        random_state=RANDOM_STATE,
        n_neighbors=n_nei,  # 控制局部与全局结构的平衡
        min_dist=min_d,  # 点在嵌入空间中的最小距离，控制聚类的紧凑度
        metric='euclidean'  # 在原始加权空间中构建图的距离
    )
    X_umap = reducer.fit_transform(X_weighted)

    print(f" - ✅ UMAP 降维完成，新维度: {X_umap.shape}")
    return X_umap, selected_features  # 返回降维后的特征


# ==========================================
# 4. HDBSCAN 聚类主流程 (优化距离函数)
# ==========================================
def run_hdbscan(X_umap, min_cluster_size):
    print("3. 🚀 正在运行 HDBSCAN 聚类...")

    # 关键修改: metric 改为 'manhattan' (L1范数)
    # 理由: UMAP 产生的嵌入空间通常更适合使用曼哈顿距离或切比雪夫距离。
    # 欧氏距离 (L2) 在高维有效，但在 UMAP 的低维流形上，L1 往往能产生更清晰的边界。
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,  # 允许算法自动设置
        metric='manhattan',  # --- 优化的距离函数 ---
        cluster_selection_method='eom',
        prediction_data=True
    )

    # 训练并预测标签
    labels = clusterer.fit_predict(X_umap)

    # 统计簇信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f" - 📊 发现簇数量: {n_clusters}")
    print(f" - 🗑️ 噪声点数量 (-1): {n_noise}")

    return labels, n_clusters, clusterer


# ==========================================
# 5. 评估、混淆矩阵绘制与结果输出 (已优化排序)
# ==========================================
def evaluate_and_save(labels, y_true, n_clusters, output_file='hdbscan_result_umap.csv'):
    print("4. 📝 正在评估结果...")

    # --- 结果保存 ---
    result_df = pd.DataFrame({
        'True_Label': y_true,
        'Predicted_Cluster': labels
    })
    result_df.to_csv(output_file, index=False)
    print(f" ✅ 详细结果已保存至: {output_file}")

    # --- 指标计算 ---
    mask = labels != -1
    sil_score = np.nan
    if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
        # 注意: 这里的 silhouette_score 使用的是降维后的空间距离 (manhattan)
        sil_score = silhouette_score(X_umap[mask], labels[mask], metric='manhattan')

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    print("-" * 30)
    print("📈 聚类评估指标:")
    print(f" - 轮廓系数 (Silhouette): {sil_score:.4f}")
    print(f" - 兰德指数 (ARI): {ari:.4f}")
    print(f" - 互信息 (NMI): {nmi:.4f}")
    print("-" * 30)

    # ==========================================
    # 🎨 核心修改：混淆矩阵排序优化
    # ==========================================
    print(" 🎨 正在绘制优化排序后的混淆矩阵...")

    # 1. 准备数据
    unique_true_labels = np.unique(y_true)
    unique_clusters = np.sort(np.unique(labels))

    # 2. 计算原始混淆矩阵 (行: 真实标签, 列: 预测簇)
    # 使用 pandas crosstab 更方便处理
    df_plot = pd.DataFrame({'True': y_true, 'Pred': labels})
    cm_raw = pd.crosstab(df_plot['True'], df_plot['Pred'], rownames=['True'], colnames=['Pred'])

    # 确保所有簇都在列中（防止某些簇未被任何标签包含导致列缺失）
    for c in unique_clusters:
        if c not in cm_raw.columns:
            cm_raw[c] = 0
    cm_raw = cm_raw.reindex(columns=sorted(cm_raw.columns))  # 按簇ID排序列

    # 3. 计算排序顺序
    # 逻辑：对于每个真实标签，找到它数量最多的那个簇（Dominant Cluster）
    # 然后根据这个 Dominant Cluster 的 ID 对真实标签进行排序
    # 这样同类设备就会聚集在一起，且尽量靠近对应的簇列

    def get_dominant_cluster(row):
        # 返回该行（真实标签）中数值最大的列名（簇ID）
        # 如果全是0，返回无穷大，排到最后
        if row.sum() == 0: return np.inf
        return row.idxmax()

    # 计算每个真实标签的主导簇
    dominant_clusters = cm_raw.apply(get_dominant_cluster, axis=1)

    # 根据主导簇对真实标签进行排序
    # 如果主导簇相同，则按标签名称排序作为次要规则
    sorted_true_labels = dominant_clusters.sort_values().index.tolist()

    # 4. 重新排列混淆矩阵
    cm_sorted = cm_raw.reindex(index=sorted_true_labels)

    # 5. 绘图
    cluster_labels = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in unique_clusters]

    plt.figure(figsize=(12, 10))  # 稍微调大一点以防标签重叠
    sns.heatmap(cm_sorted, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=cluster_labels,
                yticklabels=sorted_true_labels,  # 使用排序后的标签
                cbar_kws={'label': 'Count'})

    plt.title(f'HDBSCAN + UMAP Confusion Matrix (Sorted by Dominant Cluster)\n'
              f'Silhouette: {sil_score:.3f} | ARI: {ari:.3f} | NMI: {nmi:.3f}')
    plt.xlabel('Predicted Clusters')
    plt.ylabel('True Labels (Sorted)')
    plt.tight_layout()

    # 保存图片
    img_path = output_file.replace('.csv', '_sorted.png')  # 文件名加个 _sorted 区分
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f" ✅ 优化后的混淆矩阵图片已保存至: {img_path}")
    plt.show()
    plt.close()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    global X_umap  # 为了在 evaluate 函数中使用

    # 步骤 1: 加载数据并随机抽取15类
    # 注意: 这里为了演示抽取15类，但你的数据中可能没有那么多类，或者你想用全部类。
    # 如果报错类别不足，请调整 n_classes 参数
    X_raw, y_true_labels, sampled_classes = load_and_sample_data(
        DATA_PATH,
        n_classes=16,  # 尝试抽取15类，如果数据不足会报错
        random_state=RANDOM_STATE
    )

    # 步骤 2: 特征加权与 UMAP 降维
    X_umap, used_features = prepare_features(
        X_raw,
        IMPORTANCE_PATH,
        TOP_N_FEATURES
    )

    # 步骤 3: HDBSCAN 聚类 (使用优化后的曼哈顿距离)
    final_labels, num_clusters, cluster_model = run_hdbscan(
        X_umap,
        MIN_CLUSTER_SIZE
    )

    # 步骤 4: 评估与绘图
    evaluate_and_save(final_labels, y_true_labels, num_clusters)

    # --- 可视化 UMAP 结果 (可选) ---
    # 如果你想看数据在 UMAP 空间中的分布（按真实标签着色）
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_true_labels, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(scatter, ticks=range(len(np.unique(y_true_labels))))
    plt.title('UMAP projection of the Dataset (Colored by True Labels)')
    plt.show()