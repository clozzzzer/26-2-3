import pandas as pd
import os


def calculate_excess_lack(seq_str):
    """
    计算单个Type_Sequence的excess和lack特征
    :param seq_str: Type_Sequence列的字符串/数值
    :return: excess列表, lack元组
    """
    try:
        seq = [int(x.strip()) for x in str(seq_str).split(',') if x.strip()]
    except:
        return [], ()

    # 读取Type_Sequence的内容：去除6后的序列
    valid_seq = [num for num in seq if num != 6]
    standard_seq = [1, 2, 3, 4, 5]
    excess = []
    len_valid = len(valid_seq)

    if len_valid == 0:
        lack = tuple(standard_seq)
        return excess, lack

    if len_valid < 2:
        valid_unique = set(valid_seq)
        lack = tuple(num for num in standard_seq if num not in valid_unique)
        return excess, lack

    should_be = valid_seq[0]
    i = 0

    while i < len_valid:
        current = valid_seq[i]
        is_excess = current != should_be

        if is_excess:
            start = i
            while i < len_valid and valid_seq[i] != should_be:
                i += 1
            end = i
            excess_core = valid_seq[start:end]

            prev_elem = valid_seq[start - 1] if start > 0 else excess_core[0]
            end_elem = valid_seq[end] if end < len_valid else excess_core[-1]

            full_excess_subseq = (prev_elem,) + tuple(excess_core) + (end_elem,)
            excess.append(full_excess_subseq)
        else:
            if current < should_be:
                j = i + 1
                while j < len_valid and valid_seq[j] < should_be:
                    j += 1
                i = j
            elif current == should_be:
                should_be += 1
                i += 1
            elif current > should_be:
                should_be = current + 1
                i += 1

    valid_unique = set(valid_seq)
    lack = tuple(num for num in standard_seq if num not in valid_unique)

    return excess, lack


def calculate_range_features(range_seq_str):
    """
    基于Type_Sequence_In_Range列提取新特征
    :return: (Range_1, Range_2, Range_1_2)
    """
    try:
        seq = [int(x.strip()) for x in str(range_seq_str).split(',') if x.strip()]
    except:
        return 0, 0, 0

    range_1 = seq.count(1)
    range_2 = seq.count(2)

    range_1_2 = 0
    for i in range(len(seq) - 1):
        if seq[i] == 2 and seq[i + 1] == 1:
            range_1_2 += 1

    return range_1, range_2, range_1_2


def calculate_lack_elements(lack_tuple):
    """
    根据lack元组生成 lk_elem_1 到 lk_elem_5 的特征
    如果 lack 中包含 i，则 lk_elem_i = 1，否则为 0
    """
    if not isinstance(lack_tuple, tuple):
        return [0, 0, 0, 0, 0]

    result = []
    for i in range(1, 6):
        if i in lack_tuple:
            result.append(1)
        else:
            result.append(0)
    return result


# 1. 读取CSV文件
file_path = "./data/feature_csv_1.csv"
if not os.path.exists(file_path):
    file_path = "feature_csv_1.csv"

print(f"正在读取文件: {file_path} ...")
df = pd.read_csv(file_path)

if 'Label' not in df.columns:
    raise ValueError("错误：原CSV文件中未找到 'Label' 列。")

# 2. 计算所有新特征
# 2.1 6_Count
df['6_Count'] = df['Type_Sequence'].astype(str).str.count('6')

# 2.2 excess 和 lack (中间过程)
df[['excess', 'lack']] = df['Type_Sequence'].apply(
    lambda x: pd.Series(calculate_excess_lack(x))
)

# 2.3 【新增】根据 lack 计算 lk_elem_1 到 lk_elem_5
lack_elements = df['lack'].apply(calculate_lack_elements)
for i in range(1, 6):
    df[f'lk_elem_{i}'] = lack_elements.apply(lambda x: x[i - 1])

# 2.4 Range 相关特征
df[['Range_1', 'Range_2', 'Range_1_2']] = df['Type_Sequence_In_Range'].apply(
    lambda x: pd.Series(calculate_range_features(x))
)

# 2.5 Lack_Count
df['Lack_Count'] = df['lack'].apply(lambda x: len(x) if isinstance(x, tuple) else 0)

# 3. 定义输出列
# 【修改点】：保留 'excess'，移除 'lack'
output_columns = ['Label', 'excess', '6_Count']
output_columns += [f'lk_elem_{i}' for i in range(1, 6)]
output_columns += ['Range_1', 'Range_2', 'Range_1_2', 'Lack_Count']

# 验证列是否存在
missing_cols = [col for col in output_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"错误：以下列未在 DataFrame 中找到: {missing_cols}")

# 4. 保存文件
output_path = "./data/new_feature_csv.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df[output_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 成功！新特征文件已生成：{output_path}")
print(f"   包含列: {output_columns}")
print(f"   已排除原始 'lack' 列，保留了 'excess' 列。")

# 控制台验证预览
print("\n📊 前5行新特征文件预览:")
print(df[output_columns].head())

print("\n📈 lk_elem 特征统计 (1表示缺少该元素，0表示不缺少):")
for i in range(1, 6):
    col_name = f'lk_elem_{i}'
    count_missing = df[col_name].sum()
    print(f"   {col_name}: 缺少该元素的样本数 = {count_missing} ({count_missing / len(df) * 100:.2f}%)")