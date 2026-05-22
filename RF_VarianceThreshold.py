import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 解决中文显示问题 (根据系统自动适配，Windows一般用SimHei，Mac用Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================= 漏斗式筛选策略配置区域 =================
#A = "All_Cleaned_feature_csv0419_1_4"
A = "D0PR_All_Cleaned_feature_csv0419_1_4"
file_path = f'./data/feature/csv0419_1_4/{A}.csv'
if not os.path.exists(file_path):
    file_path = 'D0PR_csv0419_1.csv'

RESULT_DIR = './Result/Funnel_Strategy_Extended'
TARGET_CLASS_COUNT = 16  # 目标抽取的类别数量

# --- 漏斗三层筛选参数 ---
VARIANCE_THRESHOLD = 0.01
MI_TOP_K = 50
RF_FINAL_TOP_N = 20

# 创建结果目录
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

print(f"正在读取文件: {file_path} ...")

# ================= 1. 读取数据 =================
try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

df.columns = df.columns.str.strip()
if 'Label' not in df.columns:
    possible_labels = [c for c in df.columns if 'label' in c.lower() or 'type' in c.lower()]
    if possible_labels:
        df = df.rename(columns={possible_labels[0]: 'Label'})
    else:
        raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")

print(f"原始数据形状: {df.shape}")

# ================= 2. 类别采样 =================
print(f"\n🔄 正在执行类别采样策略 (目标: {TARGET_CLASS_COUNT} 类)...")
label_counts = df['Label'].value_counts()
valid_labels = label_counts[label_counts >= 2].index.tolist()

if len(valid_labels) < TARGET_CLASS_COUNT:
    print(f"❌ 错误：有效类别数量 ({len(valid_labels)}) 少于目标数量。")
    exit()

selected_labels = valid_labels[:TARGET_CLASS_COUNT]
selected_labels.sort()
df_filtered = df[df['Label'].isin(selected_labels)].reset_index(drop=True)
y = df_filtered['Label']
df = df_filtered
print(f"✅ 数据子集构建完成，样本数: {len(df)}，类别数: {y.nunique()}")

# ================= 3. 特征工程 =================
print("\n正在进行特征工程...")


def safe_eval_to_str_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '', 'nan', 'None']:
            return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1]
        parsed_list = []
        try:
            res = ast.literal_eval(x) if (x.startswith('(') or x.startswith('[')) else x
            if isinstance(res, (tuple, list)):
                parsed_list = list(res)
            else:
                parsed_list = [res]
        except Exception:
            if isinstance(x, str):
                parsed_list = [item.strip() for item in x.split(',') if item.strip()]
            else:
                parsed_list = [x]
        return [str(item) for item in parsed_list]
    return [str(x)]


feature_parts = []
feature_names_total = []

# A. 处理 Excess 特征
if 'excess' in df.columns:
    df['excess_parsed'] = df['excess'].apply(safe_eval_to_str_list)
    mlb_excess = MultiLabelBinarizer()
    excess_features = mlb_excess.fit_transform(df['excess_parsed'])
    excess_cols = [f"ex_{t}" for t in mlb_excess.classes_]
    feature_parts.append(excess_features)
    feature_names_total.extend(excess_cols)

# B. 处理序列特征
sequence_cols = ['Type_Sequence', 'Type_Sequence_In_Range']
found_seq_cols = [col for col in sequence_cols if col in df.columns]
if found_seq_cols:
    for col in found_seq_cols:
        df[f'{col}_parsed'] = df[col].apply(safe_eval_to_str_list)
        mlb_seq = MultiLabelBinarizer()
        seq_features = mlb_seq.fit_transform(df[f'{col}_parsed'])
        prefix = "seq_" if col == 'Type_Sequence' else "seq_range_"
        seq_col_names = [f"{prefix}{t}" for t in mlb_seq.classes_]
        feature_parts.append(seq_features)
        feature_names_total.extend(seq_col_names)

# C. 提取其他数值特征
exclude_cols = {'Label', 'excess', 'excess_parsed', 'Type_Sequence',
                'Type_Sequence_parsed', 'Type_Sequence_In_Range',
                'Type_Sequence_In_Range_parsed'}
lack_cols = [col for col in df.columns if 'lack' in col.lower()]
exclude_cols.update(lack_cols)
valid_numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

if valid_numeric_cols:
    numeric_data = df[valid_numeric_cols].values
    if np.isnan(numeric_data).any():
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)
    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)

# 合并所有特征
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X_full = np.hstack(feature_parts)
current_features = X_full
current_feature_names = feature_names_total
print(f"✅ 原始特征矩阵构建完成: {current_features.shape[1]} 个特征")

# ================= 4. 漏斗式特征筛选策略 =================
print("\n" + "=" * 50)
print("🚀 开始执行漏斗式三层特征筛选策略")
print("=" * 50)

# --- 第一层：方差阈值法 (粗筛) ---
print(f"\n【第一层】方差阈值筛选 (阈值: {VARIANCE_THRESHOLD})...")
selector_vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
current_features = selector_vt.fit_transform(current_features)
vt_support = selector_vt.get_support(indices=True)
current_feature_names = [current_feature_names[i] for i in vt_support]
print(f"✅ 第一层完成：剩余特征数 -> {current_features.shape[1]}")

# --- 第二层：互信息法 (精筛) ---
if current_features.shape[1] > MI_TOP_K:
    print(f"\n【第二层】互信息(MI)筛选 (保留前 {MI_TOP_K} 个)...")
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=MI_TOP_K)
    current_features = selector_mi.fit_transform(current_features, y)
    mi_support = selector_mi.get_support(indices=True)
    current_feature_names = [current_feature_names[i] for i in mi_support]
    print(f"✅ 第二层完成：剩余特征数 -> {current_features.shape[1]}")
else:
    print(f"\n【第二层】互信息(MI)筛选跳过 (当前特征数 {current_features.shape[1]} 已少于目标 {MI_TOP_K})")

# --- 第三层：随机森林重要性 (最终优化) ---
final_target = min(RF_FINAL_TOP_N, current_features.shape[1])
print(f"\n【第三层】随机森林重要性筛选 (最终保留 {final_target} 个)...")
rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
rf_temp.fit(current_features, y)

importances_temp = rf_temp.feature_importances_
top_indices = np.argsort(importances_temp)[::-1][:final_target]

X_selected = current_features[:, top_indices]
feature_names_selected = [current_feature_names[i] for i in top_indices]
print(f"✅ 第三层完成：最终保留特征数 -> {X_selected.shape[1]}")
print("=" * 50)

# ================= 5. 模型训练与评估 =================
print("\n🚀 使用最终筛选的特征进行模型训练与评估...")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n" + "=" * 50)
print(f"📊 最终分类结果报告 (基于漏斗式筛选策略)")
print("=" * 50)
print(f"准确率 (Accuracy): {acc:.4f}")
print(f"\n详细分类报告:")
print(classification_report(y_test, y_pred))

# ================= 6. 新增功能：生成混淆矩阵 =================
print("\n📊 正在生成混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix (漏斗式特征筛选)')
plt.ylabel('真实标签 (True Label)')
plt.xlabel('预测标签 (Predicted Label)')
plt.tight_layout()
cm_path = os.path.join(RESULT_DIR, 'Confusion_Matrix.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"✅ 混淆矩阵已保存至: {cm_path}")

# ================= 7. 新增功能：计算并保存特征重要性参数 =================
print("\n📊 正在计算并保存最终模型的特征重要性参数...")
final_importances = clf.feature_importances_

# 创建包含特征名和重要性得分的 DataFrame，并按重要性降序排列
importance_df = pd.DataFrame({
    'Feature_Name': feature_names_selected,
    'Importance_Score': final_importances
}).sort_values(by='Importance_Score', ascending=False)

importance_path = os.path.join(RESULT_DIR, 'Final_Feature_Importance.csv')
importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
print(f"✅ 特征重要性参数已保存至: {importance_path}")
print("\n特征重要性前 5 名预览：")
print(importance_df.head())

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance_Score', y='Feature_Name', data=importance_df, palette='viridis')
plt.title('Final Feature Importance (Top Selected Features)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
imp_plot_path = os.path.join(RESULT_DIR, 'Feature_Importance_Plot.png')
plt.savefig(imp_plot_path, dpi=300)
plt.close()
print(f"✅ 特征重要性可视化图已保存至: {imp_plot_path}")

# ================= 8. 结果保存 =================
selected_features_df = pd.DataFrame({
    'Final_Selected_Feature_Name': feature_names_selected
})
features_output_path = os.path.join(RESULT_DIR, f'Selected_Features_Funnel.csv')
selected_features_df.to_csv(features_output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 最终选中的特征列表已保存至: {features_output_path}")

result_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Correct': y_test.values == y_pred
})
output_path = os.path.join(RESULT_DIR, f'Results_Funnel.csv')
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 详细预测结果已保存至: {output_path}")

print("\n🎉 程序全部执行完毕！请前往结果文件夹查看混淆矩阵和特征重要性数据。")