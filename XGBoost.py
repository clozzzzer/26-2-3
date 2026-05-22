import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import random
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
importance_csv_path = r'.\Result/RF_All/RF_D0PR_csv0419_1_2_Cleaned_Final_Importance.csv'
data_csv_path = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'

top_n_features = 20
TRUE_LABEL_COLUMN_NAME = "Label"
RANDOM_SEED = 127
SELECTED_N_CLASSES = 25

N_ESTIMATORS = 100
MAX_DEPTH = 6
N_CLUSTERS = None


# =========================================

# ================= 中文显示修复补丁 =================
def setup_matplotlib_chinese():
    system_name = platform.system()
    plt.rcParams['axes.unicode_minus'] = False
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    print(f" ✅ 已配置 Matplotlib 中文字体 (系统: {system_name})")


setup_matplotlib_chinese()


# =========================================

def load_and_prepare_data(importance_path, data_path, top_n):
    print("1. 📂 正在读取特征重要性文件...")
    try:
        importance_df = pd.read_csv(importance_path)
        if 'Feature_Name' not in importance_df.columns:
            importance_df = pd.read_csv(importance_path, header=None,
                                        names=['Rank', 'Feature_Name', 'Importance_Score', 'Feature_Type'])

        importance_df_sorted = importance_df.sort_values(by='Importance_Score', ascending=False).head(top_n)
        selected_features = importance_df_sorted['Feature_Name'].tolist()
        weights = importance_df_sorted['Importance_Score'].values
        print(f" ✅ 成功读取，选取了 {len(selected_features)} 个特征。")
    except Exception as e:
        print(f"❌ 读取重要性文件失败: {e}")
        return None, None, None, None, None

    print("\n2. 📊 正在读取并处理原始数据...")
    try:
        full_df = pd.read_csv(data_path)

        # --- 随机抽取 N 个类别逻辑 ---
        original_total = len(full_df)
        if TRUE_LABEL_COLUMN_NAME in full_df.columns:
            all_available_labels = full_df[TRUE_LABEL_COLUMN_NAME].unique()
            total_classes = len(all_available_labels)
            random.seed(RANDOM_SEED)
            if SELECTED_N_CLASSES and SELECTED_N_CLASSES < total_classes:
                selected_classes = random.sample(list(all_available_labels), SELECTED_N_CLASSES)
                filtered_df = full_df[full_df[TRUE_LABEL_COLUMN_NAME].isin(selected_classes)]
                print(f"🔀 随机采样: 从 {total_classes} 个类别中抽取了 {SELECTED_N_CLASSES} 个。")
            else:
                filtered_df = full_df.copy()
                print(f"🔀 随机采样: 未触发抽样，使用全量数据 ({total_classes} 个类别)。")
        else:
            filtered_df = full_df.copy()
            print(f"⚠️ 未找到真实标签列，使用全量数据。")

        print(f"📊 数据规模: 原始 {original_total} 行 -> 过滤后 {len(filtered_df)} 行")

        # --- 提取真实标签和特征 ---
        true_labels = None
        if TRUE_LABEL_COLUMN_NAME in filtered_df.columns:
            true_labels = filtered_df[TRUE_LABEL_COLUMN_NAME].values

        available_features = [feat for feat in selected_features if feat in filtered_df.columns]
        if len(available_features) < len(selected_features):
            print(f"⚠️ 注意: 有 {len(selected_features) - len(available_features)} 个特征在数据中未找到。")

        X_raw = filtered_df[available_features].copy()

        # 特征加权
        weight_array = np.array([weights[selected_features.index(feat)] for feat in available_features])
        weight_array = weight_array / weight_array.max()
        X_weighted = X_raw.values * weight_array

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_weighted)

        print(f" ✅ 数据准备完成。")
        return X_scaled, filtered_df, scaler, available_features, true_labels

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return None, None, None, None, None


# ==========================================
# 🚀 XGBoost 特征转换与聚类
# ==========================================
def run_xgboost_clustering(X_train, X_test, y_train, n_clusters):
    print("\n3. 🌲 正在训练 XGBoost 并提取叶子节点特征...")

    y_train_encoded = pd.factorize(y_train)[0]
    n_classes = len(np.unique(y_train_encoded))

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train_encoded)

    print(" - 提取训练集和测试集的叶子节点索引...")
    leaves_train = model.apply(X_train)
    leaves_test = model.apply(X_test)

    print("\n4. ⚙️ 正在进行 One-Hot 编码与 KMeans 聚类...")
    encoder = OneHotEncoder()
    X_train_cluster = encoder.fit_transform(leaves_train)
    X_test_cluster = encoder.transform(leaves_test)

    final_n_clusters = n_clusters if n_clusters else n_classes
    kmeans = KMeans(n_clusters=final_n_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_test_cluster)

    return cluster_labels, X_test_cluster


# ==========================================
# 🎨 混淆矩阵绘制 (修复版 + 强制显示)
# ==========================================
def plot_confusion_matrix(true_labels, cluster_labels, true_name="真实标签", cluster_name="簇标签"):
    if true_labels is None:
        print("\n❌ 无法绘制混淆矩阵：缺少真实标签数据。")
        return

    print("\n5. 🎨 正在绘制优化排序后的混淆矩阵...")
    df_plot = pd.DataFrame({'True': true_labels, 'Pred': cluster_labels})

    # 计算混淆矩阵
    cm_raw = pd.crosstab(df_plot['True'], df_plot['Pred'], rownames=['True'], colnames=['Pred'])

    # 确保列完整
    unique_clusters = np.sort(df_plot['Pred'].unique())
    for c in unique_clusters:
        if c not in cm_raw.columns:
            cm_raw[c] = 0
    cm_raw = cm_raw.reindex(columns=sorted(cm_raw.columns))

    # 排序逻辑
    def get_dominant_cluster(row):
        if row.sum() == 0: return np.inf
        return row.idxmax()

    dominant_clusters = cm_raw.apply(get_dominant_cluster, axis=1)
    sorted_true_labels = dominant_clusters.sort_values().index.tolist()
    cm_sorted = cm_raw.reindex(index=sorted_true_labels)

    # 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_sorted, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"簇 {c}" for c in cm_sorted.columns],
                yticklabels=sorted_true_labels,
                cbar_kws={'label': '样本数量'})
    plt.title(f'XGBoost+KMeans 聚类结果混淆矩阵 (按主导簇排序)')
    plt.xlabel('预测簇 (Predicted Clusters)')
    plt.ylabel('真实标签 (True Labels - Sorted)')
    plt.tight_layout()

    # ✅ 关键修复：强制显示窗口
    plt.show()

    # 分析
    print(f"\n📊 混淆矩阵分析:")
    print(f" 真实类别数量: {len(np.unique(true_labels))}")
    print(f" 发现簇数量: {len(np.unique(cluster_labels))}")


def main():
    # 1. 加载数据
    X_processed, full_df, scaler, feature_names, true_labels = load_and_prepare_data(
        importance_csv_path, data_csv_path, top_n_features
    )
    if X_processed is None: return

    # 2. 划分训练/测试集
    # ✅ 修复：这里我们保留了原始的索引 (indices)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_processed,
        true_labels,
        np.arange(len(X_processed)),  # 保存原始索引位置
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=true_labels
    )

    print(f" - 训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 3. XGBoost 转换 + 聚类
    final_labels, X_test_cluster = run_xgboost_clustering(X_train, X_test, y_train, N_CLUSTERS)

    # 4. 评估指标
    print("\n6. 📊 正在计算评估指标...")
    ari = adjusted_rand_score(y_test, final_labels)
    nmi = normalized_mutual_info_score(y_test, final_labels)
    sil_score = silhouette_score(X_test_cluster, final_labels)

    print("-" * 30)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print("-" * 30)

    # 5. 保存结果
    print("\n7. 💾 正在保存结果...")
    # ✅ 修复：使用 idx_test (即原始数据中的行号) 来索引 full_df
    result_df = full_df.iloc[idx_test].copy()
    result_df['XGB_Cluster'] = final_labels
    output_file = 'XGBoost_Clustering_Result.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f" ✅ 结果已保存至: {output_file}")

    # 6. 绘图 (现在应该能正常弹窗了)
    plot_confusion_matrix(y_test, final_labels)


if __name__ == "__main__":
    main()