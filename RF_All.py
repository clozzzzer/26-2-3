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

A = "All_feature_csv0419_1"
file_path = f'./data/feature/csv0419_1/{A}.csv'

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

FIG_DIR = './fig/RF_All'
RESULT_DIR = './Result/RF_All'

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
print(label_counts)

# 找出样本数少于 2 的类别
rare_labels = label_counts[label_counts < 2].index.tolist()

if rare_labels:
    print(f"\n⚠️  发现 {len(rare_labels)} 个样本数少于 2 的罕见类别: {rare_labels}")
    print("这些类别无法进行分层采样，将从数据集中移除以保证模型训练稳定性。")

    # 过滤数据
    original_count = len(df)
    df = df[~df['Label'].isin(rare_labels)].reset_index(drop=True)
    removed_count = original_count - len(df)
    print(f"✅ 已移除 {removed_count} 个样本。当前数据形状: {df.shape}")

    # 重新检查
    label_counts = df['Label'].value_counts()
    if label_counts.min() < 2:
        raise ValueError("过滤后仍有类别样本数少于2，请检查数据。")
else:
    print("✅ 所有类别的样本数均 >= 2，无需过滤。")

# 2. 特征工程
print("\n正在进行特征工程...")

feature_parts = []
feature_names_total = []

# --- A. 处理 Excess 特征 ---
if 'excess' in df.columns:
    print("检测到 'excess' 列，正在解析并转换...")
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
    print(f"检测到序列特征列: {found_seq_cols}...")
    for col in found_seq_cols:
        print(f"   处理列: {col}...")
        df[f'{col}_parsed'] = df[col].apply(safe_eval_to_str_list)
        mlb_seq = MultiLabelBinarizer()
        seq_features = mlb_seq.fit_transform(df[f'{col}_parsed'])
        prefix = "seq_" if col == 'Type_Sequence' else "seq_range_"
        seq_col_names = [f"{prefix}{clean_feature_name(t)}" for t in mlb_seq.classes_]
        feature_parts.append(seq_features)
        feature_names_total.extend(seq_col_names)
        print(f"      -> {col} 生成了 {len(seq_col_names)} 个独热特征")

# --- C. 提取其他数值特征 ---
exclude_cols = {
    'Label', 'excess', 'excess_parsed',
    'Type_Sequence', 'Type_Sequence_parsed',
    'Type_Sequence_In_Range', 'Type_Sequence_In_Range_parsed'
}
lack_cols = [col for col in df.columns if 'lack' in col.lower()]
exclude_cols.update(lack_cols)
if lack_cols:
    print(f"🚫 已忽略 {len(lack_cols)} 个 lack 相关特征")

numeric_candidate_cols = [col for col in df.columns if col not in exclude_cols]
valid_numeric_cols = [col for col in numeric_candidate_cols if pd.api.types.is_numeric_dtype(df[col])]

if valid_numeric_cols:
    print(f"检测到 {len(valid_numeric_cols)} 个现成数值特征。")
    numeric_data = df[valid_numeric_cols].values
    if np.isnan(numeric_data).any():
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)
    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)

# --- D. 合并所有特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X = np.hstack(feature_parts)
y = df['Label']

print(f"\n✅ 特征矩阵构建完成:")
print(f"   总特征数: {X.shape[1]}")
print(f"   有效样本数: {X.shape[0]}")

# 3. 划分数据集与训练
# 现在可以安全地使用 stratify=y 了
test_size = 0.4
# 动态调整 test_size 以防某些类样本极少（虽然已过滤<2，但为了保险）
min_class_count = y.value_counts().min()
if min_class_count * test_size < 1:
    # 如果最小类的 40% 小于 1，则稍微减小测试集比例或确保至少取 1 个
    # 但既然已经过滤了 <2 的类，最小是 2，2*0.4 = 0.8 -> 取整为 1，sklearn 通常能处理
    pass

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("正在训练随机森林模型...")
rf_clf.fit(X_train, y_train)

# 4. 评估
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
labels_unique = sorted(df['Label'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_unique)

print(f"\n" + "=" * 50)
print("📊 模型评估结果 (RF_All)")
print("=" * 50)
print(f"测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n分类报告:")
# zero_division=0 防止因某些类在测试集未出现而报错
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# ================= 绘图部分 =================

# --- 图 1: 混淆矩阵 ---
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_unique,
            yticklabels=labels_unique)
plt.title(f'Confusion Matrix (RF_All)\nAccuracy: {accuracy:.2%}', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

save_path_1 = os.path.join(FIG_DIR, f'RF_{A}_Confusion_Matrix.png')
plt.savefig(save_path_1, dpi=300)
print(f"✅ 图片已保存: {save_path_1}")
plt.show()

# --- 图 2: 特征重要性 ---
importances = rf_clf.feature_importances_
indices_imp = np.argsort(importances)[::-1]

top_n = 20
top_indices = indices_imp[:min(top_n, len(feature_names_total))]
top_names = [feature_names_total[i] for i in top_indices]
top_scores = importances[top_indices]


def get_color(name):
    if name.startswith('ex_'): return '#1f77b4'
    if name.startswith('seq_'): return '#2ca02c'
    return '#ff7f0e'


colors = [get_color(name) for name in top_names]

plt.figure(figsize=(14, 10))
plt.barh(range(len(top_names)), top_scores[::-1], align='center', color=colors)
plt.yticks(range(len(top_names)), top_names[::-1])
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title(f'Top {len(top_names)} Feature Importances', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path_2 = os.path.join(FIG_DIR, f'RF_{A}_Feature_Importance.png')
plt.savefig(save_path_2, dpi=300)
print(f"✅ 图片已保存: {save_path_2}")
plt.show()

print("\n--- Top 15 重要特征详情 ---")
for i, idx in enumerate(top_indices):
    feat_name = feature_names_total[idx]
    score = importances[idx]
    type_tag = "[Excess]" if feat_name.startswith("ex_") else (
        "[Sequence]" if feat_name.startswith("seq_") else "[Numeric]")
    print(f"{i + 1}. {feat_name}: {score:.4f} {type_tag}")

# ================= 结果保存 =================

importance_df = pd.DataFrame({
    'Feature_Name': feature_names_total,
    'Importance_Score': importances
})
importance_df = importance_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)
importance_df.insert(0, 'Rank', range(1, len(importance_df) + 1))
importance_df['Feature_Type'] = importance_df['Feature_Name'].apply(
    lambda x: 'Excess' if x.startswith('ex_') else ('Sequence' if x.startswith('seq_') else 'Numeric'))

output_path_csv = os.path.join(RESULT_DIR, f'RF_{A}_Importance.csv')
importance_df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ 特征重要性表已保存至: {output_path_csv}")

# 保存预测结果（注意索引映射）
all_indices = np.arange(len(df))
_, test_indices, _, _ = train_test_split(all_indices, y, test_size=test_size, random_state=42, stratify=y)
test_original_index = df.index[test_indices]

result_df = pd.DataFrame({
    'True_Label': y_test,
    'Pred_Label': y_pred,
    'Correct': y_test == y_pred
}, index=test_original_index)

output_path_detail = os.path.join(RESULT_DIR, f'RF_{A}_Prediction_Result.csv')
result_df.to_csv(output_path_detail, index=True, encoding='utf-8-sig', index_label='Original_Row_Index')
print(f"✅ 预测结果详情已保存至: {output_path_detail}")

summary_file = os.path.join(RESULT_DIR, f'RF_{A}_Evaluation_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("Random Forest (All Features) Evaluation Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Total Features: {len(feature_names_total)}\n")
    f.write(f"Samples Used: {len(df)} (Removed rare classes)\n")
    f.write(f"Train Size: {len(y_train)}\n")
    f.write(f"Test Size: {len(y_test)}\n")
    if rare_labels:
        f.write(f"Removed Classes: {rare_labels}\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 程序全部执行完毕！")