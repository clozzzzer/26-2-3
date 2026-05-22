import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')

# 解决中文显示问题 (根据系统环境可能需要调整)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数 (请在此处修改)
# ==========================================
# 🔧 修改点1: 数据路径
#DATA_PATH = r'.\data/feature/csv0419_1/D0PR_csv0419_1_Cleaned_Final.csv'
DATA_PATH = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'


# ⚠️ 修改点2: 手动指定特征
# 请在下方列表中填入你想要用于聚类的列名 (必须与CSV文件中的列名完全一致)
MANUAL_FEATURES = [
    #'A_Burst_Duration',
    #'A_Burst_Rate',
    #'A_CV',
    #'A_IAT_IQR',
    #'A_IAT_Median',
    #'A_IAT_P10',
    #'A_IAT_P90',
    #'A_IAT_P95',
    #'A_IAT_Q1',
    #'A_IAT_Q3',
    #'A_IAT_Skew_Ratio',
    #'A_IAT_StdDev',
    #'A_Kurtosis',
    #'A_Mean_IAT',
    #'A_Retran_intervals',
    #'A_Silence_Avg_Duration',
    #'A_Silence_Max_Duration',
    #'A_Silence_Rate',
    #'A_Silence_Total_Duration',
    #'A_Skewness',
    #'Assoc_Time',
    #'Assoc_Time_Ratio',
    #'First_Stage_Time_Diff',
    #'Gap_23_Time',
    'Gap_23_Time_Ratio',
    'Gap_34_Time',
    'Gap_34_Time_Ratio',
    #'Gap_45_Time',
    #'Gap_45_Time_Ratio',
    'Next_Packet_After_First_Stage',
    #'Probe_Time',
    'Probe_Time_Ratio',
    #'Trans_3_3',
    #'Trans_4_5',
    #'Trans_4_6',
    #'Trans_5_1',
    #'Trans_5_6',
    #'Type_1_Count',
    #'Type_1_First_Occurrence_Time',
    #'Type_1_Last_Time',
    #'Type_2_Count',
    #'Type_2_First_Occurrence_Time',
    #'Type_2_Last_Time',
    #'Type_3_Count',
    #'Type_3_First_Occurrence_Time',
    #'Type_3_Last_Time',
    #'Type_4_Count',
    #'Type_4_First_Occurrence_Time',
    #'Type_4_Last_Time',
    #'Type_5_Count',
    #'Type_5_First_Occurrence_Time',
    #'Type_5_Last_Time',
    #'Type_6_Count',
    #'Type_6_First_Occurrence_Time',
    #'Type_6_Last_Time',
    #'Type_6_Time',
    #'Type_6_Time_Ratio',
    #'Range_1',
    'Range_2',
    'Range_1_2'
]

# ⚠️ 修改点3: HDBSCAN参数
MIN_CLUSTER_SIZE = 5  # 最小簇大小
RANDOM_STATE = 127  # 随机种子，保证每次抽取的类别一致
SELECTED_N_CLASSES = 30  # 随机抽取的类别数量

print(f"⚙️ 当前配置: 选取手动指定特征 (共 {len(MANUAL_FEATURES)} 个), HDBSCAN min_cluster_size={MIN_CLUSTER_SIZE}")


# ==========================================
# 2. 数据加载与预处理 (包含随机抽取逻辑)
# ==========================================
def load_and_sample_data(data_path, n_classes=SELECTED_N_CLASSES, random_state=RANDOM_STATE):
    print("1. 📂 正在加载数据并随机抽取类别...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 检查标签列 (假设名为 'Label')
    label_col = 'Label'
    if label_col not in df.columns:
        raise ValueError(f"未找到标签列 'Label'，请检查列名。当前列名: {df.columns.tolist()}")

    # 获取所有唯一标签
    all_labels = df[label_col].unique()
    print(f" - 原始数据集包含 {len(all_labels)} 个类别")

    # 🔑 核心修改：随机抽取 n_classes 个类别
    np.random.seed(random_state)
    selected_labels = np.random.choice(all_labels, min(n_classes, len(all_labels)), replace=False)
    print(f" - 🎲 随机抽取的 {len(selected_labels)} 个类别: {selected_labels}")

    # 筛选数据
    df_filtered = df[df[label_col].isin(selected_labels)].copy()

    # 分离特征和标签
    y_true = df_filtered[label_col].values
    X_data = df_filtered.drop(columns=[label_col])

    print(f" - ✅ 筛选后数据量: {X_data.shape[0]} 行, {X_data.shape[1]} 个特征")
    return X_data, y_true, selected_labels


# ==========================================
# 3. 特征标准化 (无加权)
# ==========================================
def prepare_features(X_data):
    print("2. ⚙️ 正在进行特征标准化 (无加权)...")

    # 🔑 核心修改：检查手动特征是否存在
    missing_cols = [c for c in MANUAL_FEATURES if c not in X_data.columns]
    if missing_cols:
        raise ValueError(f"❌ 错误: 数据中缺少以下特征列: {missing_cols}")

    # 筛选特征
    X_selected = X_data[MANUAL_FEATURES]
    print(f" - 已选特征: {MANUAL_FEATURES}")

    # 标准化 (HDBSCAN对量纲敏感，标准化是必须的)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    print(f" - ✅ 特征处理完成，形状: {X_scaled.shape}")
    return X_scaled


# ==========================================
# 4. HDBSCAN 聚类主流程
# ==========================================
def run_hdbscan(X_scaled, min_cluster_size):
    print("3. 🚀 正在运行 HDBSCAN 聚类...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # 训练并预测
    labels = clusterer.fit_predict(X_scaled)

    # 统计信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f" - 📊 发现簇数量: {n_clusters}")
    print(f" - 🗑️ 噪声点数量 (-1): {n_noise}")
    return labels, n_clusters


# ==========================================
# 5. 评估、混淆矩阵绘制与结果输出
# ==========================================
# 🔑 修改点：增加了 X_scaled 参数
def evaluate_and_save(labels, y_true, n_clusters, X_scaled, output_file='HDBSCAN_Result_NoWeight.csv'):
    print("4. 📝 正在评估结果...")

    # --- 结果保存 ---
    result_df = pd.DataFrame({
        'True_Label': y_true,
        'Predicted_Cluster': labels
    })
    result_df.to_csv(output_file, index=False)
    print(f" ✅ 详细结果已保存至: {output_file}")

    # --- 指标计算 ---
    # 1. 轮廓系数 (只对非噪声点计算)
    mask = labels != -1
    if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
        sil_score = silhouette_score(X_scaled[mask], labels[mask])
    else:
        sil_score = np.nan

    # 2. ARI 和 NMI
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    print("-" * 30)
    print("📈 聚类评估指标:")
    print(f" - 轮廓系数 (Silhouette): {sil_score:.4f}")
    print(f" - 兰德指数 (ARI): {ari:.4f}")
    print(f" - 互信息 (NMI): {nmi:.4f}")
    print("-" * 30)

    # --- 绘制混淆矩阵 ---
    print(" 🎨 正在绘制混淆矩阵...")
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)

    # 获取唯一的簇标签
    unique_clusters = np.sort(np.unique(labels))
    cluster_labels = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in unique_clusters]
    true_labels_names = le.classes_

    # 计算混淆矩阵
    cm = np.zeros((len(true_labels_names), len(unique_clusters)))
    for i, true_idx in enumerate(range(len(true_labels_names))):
        for j, cluster in enumerate(unique_clusters):
            cm[i, j] = np.sum((y_true_encoded == true_idx) & (labels == cluster))

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=cluster_labels, yticklabels=true_labels_names)
    plt.title(f'HDBSCAN Confusion Matrix (N_clusters={n_clusters})\nSilhouette: {sil_score:.3f} | ARI: {ari:.3f}')
    plt.xlabel('Predicted Clusters')
    plt.ylabel('True Labels (Ground Truth)')
    plt.tight_layout()

    # 保存图片
    img_path = output_file.replace('.csv', '.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f" ✅ 混淆矩阵图片已保存至: {img_path}")
    plt.show()
    plt.close()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 步骤 1: 加载数据并随机抽取类别
    X_raw, y_true_labels, sampled_classes = load_and_sample_data(DATA_PATH)

    # 步骤 2: 特征处理 (无加权)
    X_scaled = prepare_features(X_raw)

    # 步骤 3: HDBSCAN 聚类
    final_labels, num_clusters = run_hdbscan(X_scaled, MIN_CLUSTER_SIZE)

    # 步骤 4: 评估与绘图 (将 X_scaled 传入)
    evaluate_and_save(final_labels, y_true_labels, num_clusters, X_scaled)