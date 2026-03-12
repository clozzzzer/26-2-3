import pandas as pd
import numpy as np
import ast
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

A = 'All_feature_csv0419_1'
file_path = f'./data/feature/csv0419_1/{A}.csv'
FIG_DIR = './fig/RF_Top30'
RESULT_DIR = './Result/RF_Top30'

TOP_N_FEATURES = 30

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# ===========================================

def clean_feature_name(name):
    s = str(name)
    s = s.replace(', ', '_')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace("'", '')
    s = s.replace('"', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    return s


def safe_eval_to_str_list(x):
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '', 'nan', 'None']: return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1]

    parsed_list = []
    try:
        res = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(res, tuple):
            parsed_list = [res]
        elif isinstance(res, list):
            parsed_list = res
        else:
            parsed_list = [res] if res is not None else []
    except Exception:
        if isinstance(x, str):
            parsed_list = [item.strip() for item in x.split(',') if item.strip()]
        else:
            parsed_list = [x] if x is not None else []
    return [str(item) for item in parsed_list]


# 1. 读取数据
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

df.columns = df.columns.str.strip()
print(f"检测到的列名 ({len(df.columns)}个): {df.columns.tolist()}")
print(f"原始数据形状: {df.shape}")

if 'Label' not in df.columns:
    possible_labels = [c for c in df.columns if 'label' in c.lower() or 'type' in c.lower()]
    if possible_labels:
        df = df.rename(columns={possible_labels[0]: 'Label'})
    else:
        raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")

# 【关键修复】检查并过滤样本数过少的类别
print("\n正在检查标签分布...")
label_counts = df['Label'].value_counts()
rare_labels = label_counts[label_counts < 2].index.tolist()

if rare_labels:
    print(f"\n⚠️  发现 {len(rare_labels)} 个样本数少于 2 的罕见类别，已移除。")
    original_count = len(df)
    df = df[~df['Label'].isin(rare_labels)].reset_index(drop=True)
    print(f"✅ 已移除 {original_count - len(df)} 个样本。当前数据形状: {df.shape}")
else:
    print("✅ 所有类别的样本数均 >= 2。")

# 2. 特征工程
print("\n正在进行特征工程...")

feature_parts = []
feature_names_total = []

# --- A. 处理 Excess 特征 ---
if 'excess' in df.columns:
    df['excess_parsed'] = df['excess'].apply(safe_eval_to_str_list)
    mlb_excess = MultiLabelBinarizer()
    excess_features = mlb_excess.fit_transform(df['excess_parsed'])
    excess_cols = [f"ex_{clean_feature_name(t)}" for t in mlb_excess.classes_]
    feature_parts.append(excess_features)
    feature_names_total.extend(excess_cols)
    print(f"   -> excess 生成了 {len(excess_cols)} 个特征")

# --- B. 处理序列特征 ---
sequence_cols = ['Type_Sequence', 'Type_Sequence_In_Range']
found_seq_cols = [col for col in sequence_cols if col in df.columns]

if found_seq_cols:
    for col in found_seq_cols:
        df[f'{col}_parsed'] = df[col].apply(safe_eval_to_str_list)
        mlb_seq = MultiLabelBinarizer()
        seq_features = mlb_seq.fit_transform(df[f'{col}_parsed'])
        prefix = "seq_" if col == 'Type_Sequence' else "seq_range_"
        seq_col_names = [f"{prefix}{clean_feature_name(t)}" for t in mlb_seq.classes_]
        feature_parts.append(seq_features)
        feature_names_total.extend(seq_col_names)
        print(f"   -> {col} 生成了 {len(seq_col_names)} 个独热特征")

# --- C. 提取其他数值特征 ---
exclude_cols = {
    'Label', 'excess', 'excess_parsed',
    'Type_Sequence', 'Type_Sequence_parsed',
    'Type_Sequence_In_Range', 'Type_Sequence_In_Range_parsed'
}
lack_cols = [col for col in df.columns if 'lack' in col.lower()]
exclude_cols.update(lack_cols)

numeric_candidate_cols = [col for col in df.columns if col not in exclude_cols]
valid_numeric_cols = [col for col in numeric_candidate_cols if pd.api.types.is_numeric_dtype(df[col])]

if valid_numeric_cols:
    numeric_data = df[valid_numeric_cols].values
    if np.isnan(numeric_data).any():
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)
    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)
    print(f"   -> 数值特征生成了 {len(valid_numeric_cols)} 个特征")

# --- D. 合并所有特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X_full = np.hstack(feature_parts)
y = df['Label']

print(f"\n✅ 原始特征矩阵构建完成: {X_full.shape[1]} 个特征, {X_full.shape[0]} 个样本")

# ================= 🚀 核心修改：特征选择 (Top 30) =================
print(f"\n🔍 正在执行特征选择：仅保留最重要的 Top {TOP_N_FEATURES} 个特征...")

# 1. 训练一个临时模型来评估所有特征的重要性
temp_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
temp_rf.fit(X_full, y)
all_importances = temp_rf.feature_importances_

# 2. 获取前 N 个特征的索引
# argsort 返回的是从小到大的索引，[::-1] 反转得到从大到小
sorted_indices = np.argsort(all_importances)[::-1]
top_indices = sorted_indices[:TOP_N_FEATURES]

# 3. 筛选数据和特征名
X = X_full[:, top_indices]
feature_names_total = [feature_names_total[i] for i in top_indices]
selected_importances = all_importances[top_indices]

print(f"✅ 特征筛选完成:")
print(f"   原始特征数: {X_full.shape[1]}")
print(f"   保留特征数: {X.shape[1]}")
print(f"   移除特征数: {X_full.shape[1] - X.shape[1]}")
print(f"   Top 1 特征: {feature_names_total[0]} (Score: {selected_importances[0]:.4f})")
print(f"   Top {TOP_N_FEATURES} 特征累计重要性: {np.sum(selected_importances):.4f}")
# ===================================================================

# 3. 划分数据集与训练 (使用筛选后的 X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# 正式训练模型
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("正在训练随机森林模型 (基于 Top 30 特征)...")
rf_clf.fit(X_train, y_train)

# 4. 评估
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
labels_unique = sorted(df['Label'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_unique)

print(f"\n" + "=" * 50)
print(f"📊 模型评估结果 (RF_Top{TOP_N_FEATURES})")
print("=" * 50)
print(f"测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n分类报告:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# ================= 绘图部分 =================

# --- 图 1: 混淆矩阵 ---
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_unique,
            yticklabels=labels_unique)
plt.title(f'Confusion Matrix (RF_Top{TOP_N_FEATURES})\nAccuracy: {accuracy:.2%}', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

save_path_1 = os.path.join(FIG_DIR, f'RF_Top{TOP_N_FEATURES}_{A}_Confusion_Matrix.png')
plt.savefig(save_path_1, dpi=300)
print(f"✅ 图片已保存: {save_path_1}")
plt.show()

# --- 图 2: 特征重要性 (仅展示选用的 Top N) ---
# 由于我们只用了 Top 30，这里直接展示这 30 个的重要性排序
final_importances = rf_clf.feature_importances_
indices_imp = np.argsort(final_importances)[::-1]

# 这里 top_names 就是我们要画的全部特征
top_names = [feature_names_total[i] for i in indices_imp]
top_scores = final_importances[indices_imp]


def get_color(name):
    if name.startswith('ex_'): return '#1f77b4'
    if name.startswith('seq_'): return '#2ca02c'
    return '#ff7f0e'


colors = [get_color(name) for name in top_names]

plt.figure(figsize=(14, 10))
plt.barh(range(len(top_names)), top_scores[::-1], align='center', color=colors)
plt.yticks(range(len(top_names)), top_names[::-1])
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title(f'Top {len(top_names)} Feature Importances (Selected from 509)', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path_2 = os.path.join(FIG_DIR, f'RF_Top{TOP_N_FEATURES}_Feature_{A}_Importance.png')
plt.savefig(save_path_2, dpi=300)
print(f"✅ 图片已保存: {save_path_2}")
plt.show()

print(f"\n--- 最终选用的 {len(top_names)} 个特征详情 ---")
for i, idx in enumerate(indices_imp):
    feat_name = feature_names_total[idx]
    score = final_importances[idx]
    type_tag = "[Excess]" if feat_name.startswith("ex_") else (
        "[Sequence]" if feat_name.startswith("seq_") else "[Numeric]")
    print(f"{i + 1}. {feat_name}: {score:.4f} {type_tag}")

# ================= 结果保存 =================

# 1. 保存特征重要性表格
importance_df = pd.DataFrame({
    'Feature_Name': top_names,  # 只保存选用的特征
    'Importance_Score': top_scores
})
# 重新排序
importance_df = importance_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)
importance_df.insert(0, 'Rank', range(1, len(importance_df) + 1))
importance_df['Feature_Type'] = importance_df['Feature_Name'].apply(
    lambda x: 'Excess' if x.startswith('ex_') else ('Sequence' if x.startswith('seq_') else 'Numeric'))

output_path_csv = os.path.join(RESULT_DIR, f'RF_Top{TOP_N_FEATURES}_{A}_Importance.csv')
importance_df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ 特征重要性表已保存至: {output_path_csv}")

# 2. 保存预测结果详情
all_indices = np.arange(len(df))
_, test_indices, _, _ = train_test_split(all_indices, y, test_size=0.4, random_state=42, stratify=y)
test_original_index = df.index[test_indices]

result_df = pd.DataFrame({
    'True_Label': y_test,
    'Pred_Label': y_pred,
    'Correct': y_test == y_pred
}, index=test_original_index)

output_path_detail = os.path.join(RESULT_DIR, f'RF_Top{TOP_N_FEATURES}_{A}_Prediction_Result.csv')
result_df.to_csv(output_path_detail, index=True, encoding='utf-8-sig', index_label='Original_Row_Index')
print(f"✅ 预测结果详情已保存至: {output_path_detail}")

# 3. 保存评估摘要
summary_file = os.path.join(RESULT_DIR, f'RF_Top{TOP_N_FEATURES}_{A}_Evaluation_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(f"Random Forest (Top {TOP_N_FEATURES} Features) Evaluation Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Total Original Features: {X_full.shape[1]}\n")
    f.write(f"Selected Features: {TOP_N_FEATURES}\n")
    f.write(f"Samples Used: {len(df)}\n")
    f.write(f"Train Size: {len(y_train)}\n")
    f.write(f"Test Size: {len(y_test)}\n")
    f.write(f"\nTop 5 Features:\n")
    for i in range(min(5, len(top_names))):
        f.write(f"  {i + 1}. {top_names[i]} ({top_scores[i]:.4f})\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 程序全部执行完毕！")