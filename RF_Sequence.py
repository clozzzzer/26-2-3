import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


file_path = './data/feature/csv0419_1_feature_time_sequence.csv'

# 定义输出目录
FIG_DIR = './fig'
RESULT_DIR = './Result'

# 确保目录存在
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
    print(f"✅ 已创建图片保存目录: {FIG_DIR}")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# ===========================================

# 辅助函数：安全解析字符串列表 (用于 excess)
def safe_eval(x):
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '']: return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1]
    try:
        res = ast.literal_eval(x)
        if isinstance(res, tuple):
            return [res] if res else []
        elif isinstance(res, list):
            return res
        return []
    except Exception:
        return []


# 1. 读取数据
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

df.columns = df.columns.str.strip()
print(f"检测到的列名: {df.columns.tolist()}")
print(f"数据形状: {df.shape}")

if 'Label' not in df.columns:
    raise ValueError("❌ 错误：数据集中缺少 'Label' 列。")
if 'excess' not in df.columns:
    raise ValueError("❌ 错误：数据集中缺少 'excess' 列。")

# 2. 特征工程
print("\n正在进行特征工程...")

feature_parts = []
feature_names_total = []

# --- A. Excess 特征 (解析字符串 -> 独热编码) ---
print("检测到 'excess' 列，正在解析并转换...")
df['excess_parsed'] = df['excess'].apply(safe_eval)

mlb_excess = MultiLabelBinarizer()
excess_features = mlb_excess.fit_transform(df['excess_parsed'])
# 生成清晰的特征名
excess_cols = [f"ex_{str(t).replace(', ', '_').replace('(', '').replace(')', '')}" for t in mlb_excess.classes_]

feature_parts.append(excess_features)
feature_names_total.extend(excess_cols)
print(f"   -> excess 生成了 {len(excess_cols)} 个特征")

# --- B. 直接提取所有现成的数值特征 ---
exclude_cols = {'Label', 'excess', 'excess_parsed'}
numeric_candidate_cols = [col for col in df.columns if col not in exclude_cols]

valid_numeric_cols = []
for col in numeric_candidate_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        valid_numeric_cols.append(col)
    else:
        print(f"⚠️  跳过非数值列: '{col}' (类型: {df[col].dtype})")

if valid_numeric_cols:
    print(f"检测到 {len(valid_numeric_cols)} 个现成数值特征。")
    numeric_data = df[valid_numeric_cols].values

    if np.isnan(numeric_data).any():
        print("⚠️  数值特征中存在缺失值，填充为 0...")
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)

    feature_parts.append(numeric_data)
    feature_names_total.extend(valid_numeric_cols)

# --- C. 合并所有特征 ---
if not feature_parts:
    raise ValueError("❌ 错误：没有生成任何特征。")

X = np.hstack(feature_parts)
y = df['Label']

print(f"\n✅ 特征矩阵构建完成:")
print(f"   总特征数: {X.shape[1]}")
print(f"   样本数: {X.shape[0]}")

# 3. 划分数据集与训练
# 【关键修复】：传入 df.index 以确保 X_train, X_test 等保留原始索引信息（虽然返回的是数组，但我们可以通过 indices 追溯）
# 更简单的做法：直接对索引进行划分，然后用iloc取值
indices = np.arange(X.shape[0])
train_indices, test_indices, y_train_idx, y_test_idx = train_test_split(
    indices, y, test_size=0.5, random_state=42, stratify=y
)

# 使用索引切片获取训练集和测试集
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y.iloc[train_indices]  # 保持 Series 格式以便需要时查看索引
y_test = y.iloc[test_indices]

# 获取测试集在原始 DataFrame 中的真实索引 (用于保存结果时对应行号)
test_original_index = df.index[test_indices]

print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
print("正在训练随机森林模型...")
rf_clf.fit(X_train, y_train)

# 4. 评估与预测
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
labels_unique = sorted(df['Label'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_unique)

print(f"\n" + "=" * 40)
print("📊 模型评估结果")
print("=" * 40)
print(f"测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n分类报告:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# ================= 绘图部分 =================

# --- 图 1: 混淆矩阵 ---
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_unique,
            yticklabels=labels_unique)
plt.title(f'Confusion Matrix (Random Forest)\nAccuracy: {accuracy:.2%}', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

save_path_1 = os.path.join(FIG_DIR, 'RF_Confusion_Matrix.png')
plt.savefig(save_path_1, dpi=300)
print(f"✅ 图片已保存: {save_path_1}")
plt.show()

# --- 图 2: 特征重要性 ---
importances = rf_clf.feature_importances_
indices_imp = np.argsort(importances)[::-1]

top_n = 15
top_indices = indices_imp[:min(top_n, len(feature_names_total))]
top_names = [feature_names_total[i] for i in top_indices]
top_scores = importances[top_indices]

# 识别新数值特征以便着色
new_numeric_set = set(valid_numeric_cols)
colors = ['#ff7f0e' if name in new_numeric_set else '#1f77b4' for name in top_names]

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_names)), top_scores[::-1], align='center', color=colors)
plt.yticks(range(len(top_names)), top_names[::-1])
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title(f'Top {len(top_names)} Feature Importances\n(Orange = New Numeric Features)', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path_2 = os.path.join(FIG_DIR, 'RF_RF_Feature_Importance.png')
plt.savefig(save_path_2, dpi=300)
print(f"✅ 图片已保存: {save_path_2}")
plt.show()

# 打印详细重要性
print("\n--- Top 15 重要特征 ---")
for i, idx in enumerate(top_indices):
    feat_name = feature_names_total[idx]
    score = importances[idx]
    tag = ""
    if feat_name.startswith("ex_"):
        tag = " [Excess]"
    elif feat_name in new_numeric_set:
        tag = " [Numeric]"
    print(f"{i + 1}. {feat_name}: {score:.4f}{tag}")

# ================= 结果保存 =================

# 1. 保存特征重要性表格
importance_df = pd.DataFrame({
    'Feature_Name': feature_names_total,
    'Importance_Score': importances
})
importance_df = importance_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)
importance_df.insert(0, 'Rank', range(1, len(importance_df) + 1))

output_path_csv = os.path.join(RESULT_DIR, 'RF_Importance.csv')
importance_df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ 特征重要性表已保存至: {output_path_csv}")

# 2. 保存预测结果详情
# 【关键修复】：使用之前提取的 test_original_index 作为 DataFrame 的索引
result_df = pd.DataFrame({
    'True_Label': y_test.values,  # 转为 numpy 数组避免索引冲突
    'Pred_Label': y_pred,
    'Correct': y_test.values == y_pred
}, index=test_original_index)

output_path_detail = os.path.join(RESULT_DIR, 'RF_Prediction_Result.csv')
result_df.to_csv(output_path_detail, index=True, encoding='utf-8-sig', index_label='Original_Row_Index')
print(f"✅ 预测结果详情已保存至: {output_path_detail}")

# 3. 保存评估摘要
summary_file = os.path.join(RESULT_DIR, 'RF_Evaluation_Summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("Random Forest Evaluation Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Total Features: {len(feature_names_total)}\n")
    f.write(f"Train Size: {len(y_train)}\n")
    f.write(f"Test Size: {len(y_test)}\n\n")
    f.write("Top 5 Features:\n")
    for i in range(min(5, len(importance_df))):
        row = importance_df.iloc[i]
        f.write(f"{row['Rank']}. {row['Feature_Name']}: {row['Importance_Score']:.4f}\n")
    f.write(f"\nSaved Figures:\n")
    f.write(f"  - {save_path_1}\n")
    f.write(f"  - {save_path_2}\n")

print(f"✅ 评估摘要已保存至: {summary_file}")
print("\n🎉 程序全部执行完毕！")