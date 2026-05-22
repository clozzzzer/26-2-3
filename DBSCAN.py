import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import platform
import seaborn as sns
import random  # <--- 新增：用于随机抽样

# ================= 配置区域 =================
importance_csv_path = r'.\Result/RF_All/RF_D0PR_csv0419_1_Cleaned_Final_Importance.csv'
data_csv_path = r'.\data/feature/csv0419_1/D0PR_csv0419_1_Cleaned_Final.csv'
top_n_features = 8
target_min_samples = 7

# ⚠️ 请修改这里：你的真实标签列名是什么？
TRUE_LABEL_COLUMN_NAME = "Label"

# 🔧 新增配置：随机抽样参数
RANDOM_SEED = 127  # 随机种子，保证结果可复现
SELECTED_N_CLASSES = 30  # <--- 在这里修改：想随机选多少个类别？(例如 5, 10, 15)


# 如果设置为 None 或者 0，则使用所有类别
# =========================================

# ================= 中文显示修复补丁 =================
def setup_matplotlib_chinese():
    system_name = platform.system()
    plt.rcParams['axes.unicode_minus'] = False
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    elif system_name == "Darwin":  # Mac OS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
    else:  # Linux
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

        # --- 🔑 核心修改：随机抽取 N 个类别逻辑 ---
        original_total = len(full_df)
        selected_labels_info = "所有类别"

        if TRUE_LABEL_COLUMN_NAME in full_df.columns:
            all_available_labels = full_df[TRUE_LABEL_COLUMN_NAME].unique()
            total_classes = len(all_available_labels)

            # 设置随机种子
            random.seed(RANDOM_SEED)

            # 判断是否需要抽样
            if SELECTED_N_CLASSES and SELECTED_N_CLASSES < total_classes:
                # 随机选择 SELECTED_N_CLASSES 个标签
                selected_classes = random.sample(list(all_available_labels), SELECTED_N_CLASSES)
                # 过滤数据集
                filtered_df = full_df[full_df[TRUE_LABEL_COLUMN_NAME].isin(selected_classes)]
                selected_labels_info = f"随机选中 {SELECTED_N_CLASSES} 个: {sorted(selected_classes)}"
                print(f"🔀 随机采样: 从 {total_classes} 个类别中抽取了 {SELECTED_N_CLASSES} 个。")
            else:
                filtered_df = full_df.copy()
                selected_labels_info = f"全量数据 ({total_classes} 个类别)"
                print(f"🔀 随机采样: 未触发抽样，使用全量数据。")
        else:
            # 没有真实标签，直接使用全量数据
            filtered_df = full_df.copy()
            print(f"⚠️ 未找到真实标签列，使用全量数据。")

        print(f"📊 数据规模: 原始 {original_total} 行 -> 过滤后 {len(filtered_df)} 行")

        # --- 原有逻辑：提取真实标签和特征 ---
        true_labels = None
        if TRUE_LABEL_COLUMN_NAME in filtered_df.columns:
            true_labels = filtered_df[TRUE_LABEL_COLUMN_NAME].values
            print(f" ✅ 检测到真实标签列: {TRUE_LABEL_COLUMN_NAME} (共 {len(np.unique(true_labels))} 类)")

        # 提取用于聚类的特征
        available_features = [feat for feat in selected_features if feat in filtered_df.columns]
        X_raw = filtered_df[available_features].copy()

        weight_array = np.array([weights[selected_features.index(feat)] for feat in available_features])
        weight_array = weight_array / weight_array.max()
        X_weighted = X_raw.values * weight_array

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_weighted)

        print(f" ✅ 数据准备完成。")

        # 返回时也返回 filtered_df 用于后续保存结果
        return X_scaled, filtered_df, scaler, available_features, true_labels

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return None, None, None, None, None


def scan_eps_and_cluster(X_scaled, min_samples):
    print("\n3. 🔍 正在扫描最佳 EPS 范围...")
    best_eps = None
    best_score = -1
    best_labels = None
    best_n_clusters = 0

    for eps in np.arange(0.5, 15.5, 0.5):
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = list(labels).count(-1)
        noise_ratio = noise_count / len(labels)

        if n_clusters > 1 and noise_ratio < 0.3:
            score = n_clusters * (1 - noise_ratio)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels
                best_n_clusters = n_clusters

    if best_eps is None:
        best_eps = 1.5
        clustering = DBSCAN(eps=best_eps, min_samples=min_samples)
        best_labels = clustering.fit_predict(X_scaled)
        best_n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)

    return best_labels, best_eps, best_n_clusters


def plot_confusion_matrix(true_labels, cluster_labels, true_name="真实标签", cluster_name="簇标签"):
    """ 绘制混淆矩阵热力图 """
    if true_labels is None:
        print("\n❌ 无法绘制混淆矩阵：缺少真实标签数据。")
        return

    print("\n5. 🎨 正在绘制混淆矩阵...")
    # 创建一个 DataFrame 用于展示映射关系
    # 注意：DBSCAN 的 -1 噪声点在这里被视为一个特殊的“簇”
    mapping_df = pd.DataFrame({
        true_name: true_labels,
        cluster_name: cluster_labels
    })

    # 生成交叉表 (Confusion Matrix)
    # index: 真实标签, columns: 聚类标签
    matrix = pd.crosstab(mapping_df[true_name], mapping_df[cluster_name], margins=False)

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'簇 {i}' for i in matrix.columns],
                yticklabels=matrix.index)
    plt.title(f'真实标签 vs DBSCAN 簇 分布图\n(混淆矩阵)')
    plt.ylabel(true_name)
    plt.xlabel('DBSCAN ' + cluster_name)
    plt.tight_layout()
    plt.show()

    # --- 简单的分析 ---
    print(f"\n📊 混淆矩阵分析:")
    print(f" 真实类别数量: {len(np.unique(true_labels))}")
    print(f" 发现簇数量: {len(np.unique(cluster_labels[cluster_labels != -1]))}")
    print(f" 噪声点数量: {np.sum(cluster_labels == -1)}")


def main():
    # 1. 加载数据 (现在包含 true_labels)
    X_processed, full_df, scaler, feature_names, true_labels = load_and_prepare_data(
        importance_csv_path, data_csv_path, top_n_features
    )
    if X_processed is None:
        return

    # 2. 聚类
    labels, best_eps, n_clusters = scan_eps_and_cluster(X_processed, target_min_samples)

    # 3. 保存结果
    print("\n4. 💾 正在生成结果报告...")
    result_df = full_df.copy()  # 这里的 full_df 已经是过滤后的数据
    result_df['DBSCAN_Cluster'] = labels
    result_df = result_df.sort_values(by='DBSCAN_Cluster')
    output_file = 'DBSCAN_Clustering_Result.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f" ✅ 结果已保存至: {output_file}")

    # 4. 统计报告
    print("\n" + "=" * 60)
    print("📊 最终聚类结果报告")
    print("=" * 60)
    noise_count = list(labels).count(-1)
    noise_ratio = noise_count / len(labels)
    print(f"最佳参数: eps={best_eps}, min_samples={target_min_samples}")
    print(f"发现的簇数量: {n_clusters}")
    print(f"噪声点数量: {noise_count} ({noise_ratio * 100:.1f}%)")

    # 5. 绘制混淆矩阵 (如果有的话)
    # 注意：这里传入原始的 true_labels 和 DBSCAN 的 labels
    plot_confusion_matrix(true_labels, labels, TRUE_LABEL_COLUMN_NAME, "Cluster")

    # 6. 原有的可视化 (散点图/雷达图)
    if n_clusters > 0:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        cmap = plt.colormaps.get_cmap('tab10')
        colors = ['lightgray' if l == -1 else cmap(l % 10) for l in labels]
        ax[0].scatter(X_processed[:, 0], X_processed[:, 1], c=colors, s=50, alpha=0.7, edgecolors='k')
        ax[0].set_xlabel(f'{feature_names[0]} (Scaled)')
        ax[0].set_ylabel(f'{feature_names[1]} (Scaled)')
        ax[0].set_title(f'DBSCAN 聚类分布 (eps={best_eps})')
        ax[0].grid(True, alpha=0.3)

        # 雷达图代码保持不变...
        if n_clusters > 1:  # 避免只有一个簇时报错
            radar_clusters = [i for i in range(min(3, n_clusters))]
            angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
            angles += angles[:1]
            for cluster_id in radar_clusters:
                cluster_data = result_df[result_df['DBSCAN_Cluster'] == cluster_id][feature_names]
                means = cluster_data.mean().values.tolist()
                means += means[:1]
                ax[1].plot(angles, means, 'o-', linewidth=2, label=f'簇 {cluster_id}')
                ax[1].fill(angles, means, alpha=0.15)
            ax[1].set_xticks(angles[:-1])
            ax[1].set_xticklabels(feature_names, fontsize=8)
            ax[1].set_title(f'簇特征画像 (雷达图)')
            ax[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax[1].grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()