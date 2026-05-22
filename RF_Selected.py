import pandas as pd
import numpy as np
import ast
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

file_path = 'data/feature/csv0419_1_4/D0PR_csv0419_1_Cleaned_Final.csv'
FIG_DIR = './fig/RF_Selected'
RESULT_DIR = './Result/RF_Selected'
TARGET_CLASS_COUNT = 16  # ⬅️ 目标抽取的类别数量
TOP_N_FEATURES = 20 # ⬅️ 特征数量

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


# ================= 1. 读取数据 =================
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

df.columns = df.columns.str.strip()

if 'Label' not in df.columns:
    possible_labels = [c for c in df.columns if 'label' in c.lower() or 'type' in c.lower()]
    if possible_labels:
        df = df.rename(columns={possible_labels[0]: 'Label'})
    else:
        raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")

print(f"原始数据形状: {df.shape}")
print(f"原始类别总数: {df['Label'].nunique()}")

# ================= 2. 🎯 核心修改：随机抽取 15 个类 =================
print(f"\n🔄 正在执行类别采样策略 (目标: {TARGET_CLASS_COUNT} 类)...")

# A. 统计样本分布
label_counts = df['Label'].value_counts()

# B. 过滤掉样本数 < 2 的类别 (无法进行 train/test 分割)
valid_labels = label_counts[label_counts >= 2].index.tolist()
removed_rare = label_counts[label_counts < 2].index.tolist()

if removed_rare:
    print(f"⚠️  已移除 {len(removed_rare)} 个样本数不足的罕见类别。")

if len(valid_labels) < TARGET_CLASS_COUNT:
    print(f"❌ 错误：有效类别数量 ({len(valid_labels)}) 少于目标数量 ({TARGET_CLASS_COUNT})。")
    print(f"   有效类别列表: {valid_labels}")
    raise ValueError("无法抽取足够的类别。")

# C. 随机抽取
selected_labels = random.sample(valid_labels, TARGET_CLASS_COUNT)
selected_labels.sort()  # 排序以便展示

print(f"✅ 成功随机抽取 {TARGET_CLASS_COUNT} 个类别:")
for lbl in selected_labels:
    count = label_counts[lbl]
    print(f"   - {lbl} (样本数: {count})")

# D. 过滤数据集
df_filtered = df[df['Label'].isin(selected_labels)].reset_index(drop=True)
y = df_filtered['Label']

print(f"\n📉 数据子集构建完成:")
print(f"   原始样本数: {len(df)} -> 子集样本数: {len(df_filtered)}")
print(f"   原始类别数: {df['Label'].nunique()} -> 子集类别数: {y.nunique()}")

# 更新 df 为过滤后的数据，后续步骤均基于此子集
df = df_filtered

# ================= 3. 特征工程 =================
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

# --- D. 合并所有特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X_full = np.hstack(feature_parts)
print(f"✅ 原始特征矩阵构建完成: {X_full.shape[1]} 个特征")

# ================= 4. 特征选择 (Top 30) =================
print(f"\n🔍 正在基于子集数据执行特征选择 (Top {TOP_N_FEATURES})...")

# 使用随机森林进行特征重要性评估
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_selector.fit(X_full, y)
all_importances = rf_selector.feature_importances_

sorted_indices = np.argsort(all_importances)[::-1]
top_indices = sorted_indices[:TOP_N_FEATURES]

X_selected = X_full[:, top_indices]
feature_names_selected = [feature_names_total[i] for i in top_indices]

print(f"✅ 特征筛选完成: 保留 {len(feature_names_selected)} 个特征")

# ================= 5. 划分训练集与测试集 =================
print("\n正在划分训练集和测试集 (Test Size=0.3)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.5, random_state=42, stratify=y
)
print(f"   训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")

# ================= 6. 模型训练 (Random Forest) =================
print("\n正在训练随机森林分类器...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # 处理可能的类别不平衡
)
clf.fit(X_train, y_train)

# ================= 7. 模型评估 =================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n" + "=" * 50)
print(f"📊 分类结果报告 (基于随机抽取的 {TARGET_CLASS_COUNT} 类)")
print("=" * 50)
print(f"准确率 (Accuracy): {acc:.4f}")
print(f"\n详细分类报告:")
print(classification_report(y_test, y_pred))

# ================= 8. 绘图部分 =================

# --- 图 1: 混淆矩阵 ---
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.title(f'Confusion Matrix (Random {TARGET_CLASS_COUNT} Classes)\nAccuracy: {acc:.2%}', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

save_path_cm = os.path.join(FIG_DIR, f'RF_Random{TARGET_CLASS_COUNT}_ConfusionMatrix.png')
plt.savefig(save_path_cm, dpi=300)
print(f"✅ 混淆矩阵已保存: {save_path_cm}")
plt.show()

# --- 图 2: Top 30 特征重要性 ---
indices_imp = np.argsort(clf.feature_importances_)[::-1]
top_names = [feature_names_selected[i] for i in indices_imp]
top_scores = clf.feature_importances_[indices_imp]


def get_color(name):
    if name.startswith('ex_'): return '#1f77b4'
    if name.startswith('seq_'): return '#2ca02c'
    return '#ff7f0e'


colors = [get_color(name) for name in top_names]

plt.figure(figsize=(14, 8))
plt.barh(range(len(top_names)), top_scores[::-1], align='center', color=colors)
plt.yticks(range(len(top_names)), top_names[::-1])
plt.xlabel('Feature Importance', fontsize=12)
plt.title(f'Top {len(top_names)} Features (Random {TARGET_CLASS_COUNT} Classes)', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path_feat = os.path.join(FIG_DIR, f'RF_Random{TARGET_CLASS_COUNT}_Features.png')
plt.savefig(save_path_feat, dpi=300)
print(f"✅ 特征重要性图已保存: {save_path_feat}")
plt.show()

# ================= 9. 结果保存 =================
result_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Correct': y_test.values == y_pred
})
output_path = os.path.join(RESULT_DIR, f'RF_Random{TARGET_CLASS_COUNT}_Results.csv')
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 详细预测结果已保存至: {output_path}")

summary_file = os.path.join(RESULT_DIR, f'RF_Random{TARGET_CLASS_COUNT}_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(f"Random Forest Classification Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Target Class Count: {TARGET_CLASS_COUNT}\n")
    f.write(f"Selected Classes: {selected_labels}\n")
    f.write(f"Total Samples Used: {len(df)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"\nTop 20 Features:\n")
    for i in range(min(20, len(top_names))):
        f.write(f"  {i + 1}. {top_names[i]} ({top_scores[i]:.4f})\n")

print(f"✅ 摘要信息已保存至: {summary_file}")
print("\n🎉 程序执行完毕！")