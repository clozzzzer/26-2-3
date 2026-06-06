import pandas as pd

# 1. 配置文件路径
input_file = 'data/feature/csv0419_1_5/All_Cleaned_feature_csv0419_1_5.csv'
#IMPORTANCE_PATH = r'./Result/RF_All\RF_All_Cleaned_feature_csv0419_1_4_Importance.csv'
IMPORTANCE_PATH = r'./Result/Funnel_Strategy_Extended/Final_Feature_Importance.csv'
output_file = 'data/feature/csv0419_1_5/D0PR_All_Cleaned_feature_csv0419_1_5.csv'

# --- 新增配置：定义标签列名 ---
# 请根据你原始CSV文件的实际列名修改这里
LABEL_COLUMN = "Label"
# 如果上面的名字不对，请改成你文件里实际的标签列名，例如：
# LABEL_COLUMN = "label"
# LABEL_COLUMN = "target"

print(f"1. 正在读取重要性文件以确定保留的特征...")
try:
    # 2. 读取重要性文件 (指定header=impo0，因为你的文件有表头)
    importance_df = pd.read_csv(IMPORTANCE_PATH, header=0)

    # 提取前10个特征名 (从 'Feature_Name' 列提取)
    top_10_features = importance_df['Feature_Name'].head(20).tolist()

    print(f"✅ 成功读取重要性文件，前10个重要特征为:")
    for i, feature in enumerate(top_10_features, 1):
        print(f"   {i}. {feature}")

except Exception as e:
    print(f"❌ 读取重要性文件失败: {e}")
    exit()

print(f"\n2. 正在处理原始数据文件...")
try:
    # 3. 读取原始CSV数据
    df = pd.read_csv(input_file, on_bad_lines='skip', engine='python')
    print(f"✅ 读取成功，原始列数: {len(df.columns)}")

    # 4. 构建最终保留的列列表 (关键修改点：确保标签在最前面)
    columns_to_keep = []

    # --- 第一步：先放入标签列 ---
    if LABEL_COLUMN in df.columns:
        columns_to_keep.append(LABEL_COLUMN)
        print(f"\n➕ 已将标签列 '{LABEL_COLUMN}' 置于首位")
    else:
        print(f"\n⚠️  警告: 原始数据中未找到标签列 '{LABEL_COLUMN}'。请检查列名是否正确！")
        # 如果没有找到标签，依然继续执行，但只保留特征

    # --- 第二步：再依次放入前10个重要特征 ---
    for col in top_10_features:
        if col in df.columns:
            columns_to_keep.append(col)
        # else: 特征不存在则跳过

    # 5. 筛选数据
    df_cleaned = df[columns_to_keep]

    print(f"\n3. 📊 结果统计:")
    print(f"   总保留列数: {len(df_cleaned.columns)}")
    print(f"   最终列名 (顺序): {list(df_cleaned.columns)}")

    # 6. 保存结果
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 清洗完成，已保存至: {output_file}")

except Exception as e:
    print(f"❌ 处理原始文件失败: {e}")