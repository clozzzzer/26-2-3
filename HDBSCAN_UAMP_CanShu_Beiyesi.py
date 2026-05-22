import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import warnings
import matplotlib.pyplot as plt
import umap

# 引入 Optuna 库
import optuna

# 忽略警告
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数
# ==========================================
IMPORTANCE_PATH = r'./Result/RF_All\RF_All_Cleaned_feature_csv0419_1_2_Importance.csv'
DATA_PATH = r'.\data/feature/csv0419_1_2/All_Cleaned_feature_csv0419_1_2.csv'
TOP_N_FEATURES = 8
RANDOM_STATE = 42

# 综合评分权重
WEIGHT_ARI = 0.7
WEIGHT_SIL = 0.2
WEIGHT_COV = 0.1

# Optuna 优化设置
N_TRIALS = 200  # 优化的迭代次数（即尝试多少组参数）


# ==========================================
# 2. 数据加载与预处理 (保持不变)
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
# 3. Optuna 贝叶斯优化核心部分
# ==========================================

# 全局变量，用于在目标函数和主程序间传递数据
global_X_weighted = None
global_y_true = None
global_best_result = None


# --- 定义 Optuna 的目标函数 ---
def objective(trial):
    """
    Optuna 会自动调用这个函数。
    我们需要在函数内部用 trial.suggest_* 来定义参数空间。
    """
    # A. 定义参数的搜索空间 (连续空间)
    n_components = trial.suggest_int('n_components', 2, 5)
    n_neighbors = trial.suggest_int('n_neighbors', 5, 10)
    min_dist = trial.suggest_float('min_dist', 0.0, 0.3)
    min_cluster_size = trial.suggest_int('min_cluster_size', 2, 15)

    try:
        # B. 执行 UMAP 降维
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=RANDOM_STATE
        )
        X_umap = reducer.fit_transform(global_X_weighted)

        # C. 执行 HDBSCAN 聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='manhattan',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(X_umap)

        # D. 计算综合评分指标
        mask = labels != -1
        coverage = np.sum(mask) / len(labels) if len(labels) > 0 else 0
        ari = adjusted_rand_score(global_y_true, labels)

        sil = -1.0
        # 只有在有非噪声点且簇数量大于1时才计算轮廓系数
        if coverage > 0 and len(set(labels[labels != -1])) > 1:
            sil = silhouette_score(X_umap[mask], labels[mask], metric='manhattan')

        # 计算综合得分
        norm_ari = (ari + 1) / 2
        norm_sil = (sil + 1) / 2
        total_score = (WEIGHT_ARI * norm_ari) + (WEIGHT_SIL * norm_sil) + (WEIGHT_COV * coverage)

        # 记录历史最佳结果，供最终输出使用
        global global_best_result
        if global_best_result is None or total_score > global_best_result['score']:
            global_best_result = {
                'score': total_score,
                'params': {
                    'UMAP_n_components': n_components,
                    'UMAP_n_neighbors': n_neighbors,
                    'UMAP_min_dist': min_dist,
                    'HDBSCAN_min_cluster_size': min_cluster_size
                },
                'labels': labels.copy(),
                'X_umap': X_umap.copy()
            }

        # 返回负值，因为 Optuna 默认寻找最小值
        return -total_score

    except Exception as e:
        # 如果参数组合非法，返回一个极大的惩罚值
        return 10.0


def run_optuna_optimization(X_weighted, y_true):
    print("\n🚀 开始 Optuna 贝叶斯优化搜索...")
    print(f"计划迭代次数: {N_TRIALS} 次\n")

    global global_X_weighted, global_y_true, global_best_result
    global_X_weighted = X_weighted
    global_y_true = y_true
    global_best_result = None

    # 创建 Optuna 研究对象 (direction="minimize" 表示寻找最小值)
    study = optuna.create_study(direction="minimize")

    # 执行优化
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("🏆 Optuna 优化搜索结束！")
    print(f"最佳综合得分: {-study.best_value:.4f}")
    print("最佳参数组合:")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
    print("=" * 60)

    return global_best_result


# ==========================================
# 4. 评估与可视化 (保持不变)
# ==========================================
def evaluate_and_save(result_dict, output_file='hdbscan_optuna_result.csv'):
    if result_dict is None:
        print("❌ 优化过程未产生有效结果。")
        return

    labels = result_dict['labels']
    X_umap = result_dict['X_umap']
    best_params = result_dict['params']

    print("4. 📝 正在生成最终报告...")
    mask = labels != -1
    coverage = np.sum(mask) / len(labels)
    ari = adjusted_rand_score(global_y_true, labels)
    nmi = normalized_mutual_info_score(global_y_true, labels)

    sil = -1
    if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
        sil = silhouette_score(X_umap[mask], labels[mask], metric='manhattan')

    print(f"\n📈 最终聚类性能报告:")
    print(f" - 综合得分: {result_dict['score']:.4f}")
    print(f" - 准确率 (ARI): {ari:.4f}")
    print(f" - 互信息 (NMI): {nmi:.4f}")
    print(f" - 轮廓系数 (Sil): {sil:.4f}")
    print(f" - 聚类覆盖率: {coverage:.4f}")
    print("-" * 40)

    result_df = pd.DataFrame({'True_Label': global_y_true, 'Predicted_Cluster': labels})
    result_df.to_csv(output_file, index=False)

    # 绘图 (取前两维)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='Spectral', s=5)
    plt.title(f'Optuna Optimized Clustering\nARI: {ari:.3f} | Sil: {sil:.3f}')
    plt.colorbar()
    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        X_raw, y_true_labels = load_and_sample_data(DATA_PATH)
        X_weighted = prepare_features(X_raw, IMPORTANCE_PATH, TOP_N_FEATURES)

        # 运行 Optuna 贝叶斯优化
        best_result = run_optuna_optimization(X_weighted, y_true_labels)

        # 评估并保存结果
        evaluate_and_save(best_result)

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()