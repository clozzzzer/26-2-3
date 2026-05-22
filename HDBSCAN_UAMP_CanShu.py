import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import umap

# 忽略警告
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数
# ==========================================
#IMPORTANCE_PATH = r'./Result/RF_All\RF_All_Cleaned_feature_csv0419_1_3_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1_3/All_Cleaned_feature_csv0419_1_3.csv'
#IMPORTANCE_PATH = r'./Result/RF_All\RF_All_Cleaned_feature_csv0419_1_4_Importance.csv'
#DATA_PATH = r'.\data/feature/csv0419_1_4/All_Cleaned_feature_csv0419_1_4.csv'
IMPORTANCE_PATH = r'Result/Funnel_Strategy_Extended/Final_Feature_Importance.csv'
DATA_PATH = r'data/feature/csv0419_1_4/D0PR_All_Cleaned_feature_csv0419_1_4.csv'
TOP_N_FEATURES = 8
RANDOM_STATE = 42

# --- 参数搜索范围 ---
UMAP_N_NEIGHBORS = list(range(2, 13))
UMAP_MIN_DIST = np.arange(0, 0.05, 0.01).tolist()
HDBSCAN_MIN_CLUSTER_SIZE = list(range(2, 7))
UMAP_N_COMPONENTS = list(range(2, 5))  # 搜索 2, 3, 4

# --- 综合评分配置 ---
WEIGHT_ARI = 0.9
WEIGHT_SIL = 0.05
WEIGHT_COV = 0.05
print(f"⚙️ 综合评分权重: ARI={WEIGHT_ARI}, Silhouette={WEIGHT_SIL}, Coverage={WEIGHT_COV}")


# ==========================================
# 2. 数据加载与预处理
# ==========================================
def load_and_sample_data(data_path, n_classes=25, random_state=42):
    print("1. 📂 正在加载数据...")
    df = pd.read_csv(data_path)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    if label_col not in df.columns:
        raise ValueError("未找到标签列")
    all_labels = df[label_col].unique()
    n_classes = min(n_classes, len(all_labels))
    np.random.seed(random_state)
    selected_labels = np.random.choice(all_labels, n_classes, replace=False)
    df_filtered = df[df[label_col].isin(selected_labels)].copy()
    y_true = df_filtered[label_col].values
    X_data = df_filtered.drop(columns=[label_col])
    print(f" - ✅ 数据加载完成: {X_data.shape[0]} 行")
    return X_data, y_true


def prepare_features(X_data, importance_path, top_n):
    print("2. ⚙️ 正在特征加权...")
    imp_df = pd.read_csv(importance_path)
    if 'Feature_Name' not in imp_df.columns:
        imp_df.columns = ['Feature_Name', 'Importance_Score']
    imp_df_sorted = imp_df.sort_values(by='Importance_Score', ascending=False).head(top_n)
    selected_features = imp_df_sorted['Feature_Name'].tolist()
    selected_features = [c for c in selected_features if c in X_data.columns]
    if not selected_features:
        selected_features = X_data.columns[:top_n].tolist()
    weights = imp_df_sorted.set_index('Feature_Name').loc[selected_features, 'Importance_Score'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data[selected_features])
    X_weighted = X_scaled * weights
    return X_weighted


# ==========================================
# 3. 核心：综合评分参数搜索 (修复版)
# ==========================================
def find_best_parameters(X_weighted, y_true):
    print("\n🔍 开始多维参数搜索 (综合评分法)...")

    # --- 【关键修改】强制执行一次保底参数 ---
    print("🔧 正在执行保底参数 (n_components=2, neighbors=5, dist=0.1, size=5)...")
    try:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                            n_neighbors=5, min_dist=0.1, metric='euclidean')
        X_umap = reducer.fit_transform(X_weighted)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='manhattan',
                                    cluster_selection_method='eom', prediction_data=True)
        labels = clusterer.fit_predict(X_umap)

        # 计算保底分数
        mask = labels != -1
        coverage = np.sum(mask) / len(labels) if len(labels) > 0 else 0
        ari = adjusted_rand_score(y_true, labels)
        sil = -1.0
        if coverage > 0 and len(set(labels[labels != -1])) > 1:
            sil = silhouette_score(X_umap[mask], labels[mask], metric='manhattan')

        # 初始化最佳结果为保底结果
        norm_ari = (ari + 1) / 2
        norm_sil = (sil + 1) / 2
        best_total_score = (WEIGHT_ARI * norm_ari) + (WEIGHT_SIL * norm_sil) + (WEIGHT_COV * coverage)
        best_params = {
            'UMAP_n_components': 2,
            'UMAP_n_neighbors': 5,
            'UMAP_min_dist': 0.1,
            'HDBSCAN_min_cluster_size': 5
        }
        best_labels = labels.copy()
        best_X_umap = X_umap.copy()
        print(f" [保底成功] 总分: {best_total_score:.4f}, ARI={ari:.3f}")

    except Exception as e:
        print(f" [警告] 保底参数失败，将使用空初始化: {e}")
        best_total_score = -1
        best_params = {}
        best_labels = None
        best_X_umap = None

    results_log = []

    # 计算总步数 (包含 n_components)
    total_steps = len(UMAP_N_NEIGHBORS) * len(UMAP_MIN_DIST) * len(HDBSCAN_MIN_CLUSTER_SIZE) * len(UMAP_N_COMPONENTS)
    current_step = 0

    # --- 开始搜索循环 ---
    for n_components in UMAP_N_COMPONENTS:
        for n_neighbors in UMAP_N_NEIGHBORS:
            for min_dist in UMAP_MIN_DIST:
                for min_cluster_size in HDBSCAN_MIN_CLUSTER_SIZE:
                    current_step += 1
                    if current_step % 100 == 0:
                        print(f" - 进度: {current_step}/{total_steps}...")

                    try:
                        # 执行 UMAP + HDBSCAN
                        reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE,
                                            n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean')
                        X_umap = reducer.fit_transform(X_weighted)

                        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='manhattan',
                                                    cluster_selection_method='eom', prediction_data=True)
                        labels = clusterer.fit_predict(X_umap)

                        # 计算指标
                        noise_count = np.sum(labels == -1)
                        coverage = (len(labels) - noise_count) / len(labels)
                        ari = adjusted_rand_score(y_true, labels)

                        sil = -1.0
                        if coverage > 0 and len(set(labels[labels != -1])) > 1:
                            sil = silhouette_score(X_umap[labels != -1], labels[labels != -1], metric='manhattan')

                        # 计算综合得分
                        norm_ari = (ari + 1) / 2
                        norm_sil = (sil + 1) / 2
                        norm_cov = coverage
                        total_score = (WEIGHT_ARI * norm_ari) + (WEIGHT_SIL * norm_sil) + (WEIGHT_COV * norm_cov)

                        # 更新最佳结果
                        if total_score > best_total_score:
                            best_total_score = total_score
                            best_params = {
                                'UMAP_n_components': n_components,
                                'UMAP_n_neighbors': n_neighbors,
                                'UMAP_min_dist': min_dist,
                                'HDBSCAN_min_cluster_size': min_cluster_size
                            }
                            best_labels = labels.copy()
                            best_X_umap = X_umap.copy()
                            print(f"\n [🏆 新纪录] 总分: {total_score:.4f}")
                            print(f" 参数: {best_params}")
                            print(f" 指标: ARI={ari:.3f}, Sil={sil:.3f}, Cov={coverage:.3f}")

                    except Exception as e:
                        # print(f"参数组合失败: {e}") # 调试时可开启
                        continue

    # --- 搜索结束 ---
    if best_labels is None:
        print("❌ 严重错误：所有参数组合均失败，无法生成结果。")
        # 创建一个全是噪声的假标签作为最后的最后手段
        best_labels = np.array([-1] * X_weighted.shape[0])
        best_X_umap = np.zeros((X_weighted.shape[0], 2))

    print("\n" + "=" * 60)
    print("🏆 搜索结束！最佳参数组合:")
    for k, v in best_params.items():
        print(f" {k}: {v}")
    print("=" * 60)
    return best_X_umap, best_labels, best_params


# ==========================================
# 4. 评估与可视化 (增加防御)
# ==========================================
def evaluate_and_save(labels, y_true, X_umap, output_file='hdbscan_result_final.csv'):
    print("4. 📝 正在生成最终报告...")

    # --- 【防御】检查输入 ---
    if labels is None or len(labels) == 0:
        print("❌ 输入数据无效 (labels 为空或 None)，正在生成空报告...")
        # 创建空文件或默认输出
        with open(output_file, 'w') as f:
            f.write("Error: No valid labels generated.\n")
        return

    mask = labels != -1
    coverage = np.sum(mask) / len(labels)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    sil = -1
    if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
        sil = silhouette_score(X_umap[mask], labels[mask], metric='manhattan')

    print(f"\n📈 最终聚类性能报告:")
    print(f" - 准确率 (ARI): {ari:.4f}")
    print(f" - 互信息 (NMI): {nmi:.4f}")
    print(f" - 轮廓系数 (Sil): {sil:.4f}")
    print(f" - 聚类覆盖率: {coverage:.4f}")
    print("-" * 40)

    result_df = pd.DataFrame({'True_Label': y_true, 'Predicted_Cluster': labels})
    result_df.to_csv(output_file, index=False)

    # 绘图 (取前两维)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='Spectral', s=5)
    plt.title(f'Best Clustering Result\nARI: {ari:.3f} | Sil: {sil:.3f}')
    plt.colorbar()
    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        X_raw, y_true_labels = load_and_sample_data(DATA_PATH)
        X_weighted = prepare_features(X_raw, IMPORTANCE_PATH, TOP_N_FEATURES)
        X_best_umap, best_labels, best_params = find_best_parameters(X_weighted, y_true_labels)
        evaluate_and_save(best_labels, y_true_labels, X_best_umap)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()