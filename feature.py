import os
import pandas as pd
import numpy as np

# ==========================================
# 配置区域
# ==========================================
# 请确认此路径为您实际的数据根目录
BASE_PATH = "./data/csv0419_1"
OUTPUT_FILE = "./data/feature/csv0419_1/All_feature_csv0419_1.csv"


# ==========================================
# 第一部分：序列特征提取函数
# ==========================================

def calculate_excess_lack(seq_str):
    """
    计算 Type_Sequence 的 excess 和 lack 特征
    返回: excess (列表), lack (元组)
    """
    try:
        if pd.isna(seq_str) or str(seq_str).strip() == '':
            return [], ()
        seq = [int(x.strip()) for x in str(seq_str).split(',') if x.strip()]
    except Exception:
        return [], ()

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
    计算 Type_Sequence_In_Range 的 Range_1, Range_2, Range_1_2 特征
    """
    try:
        if pd.isna(range_seq_str) or str(range_seq_str).strip() == '':
            return 0, 0, 0
        seq = [int(x.strip()) for x in str(range_seq_str).split(',') if x.strip()]
    except Exception:
        return 0, 0, 0

    range_1 = seq.count(1)
    range_2 = seq.count(2)

    range_1_2 = 0
    for i in range(len(seq) - 1):
        if seq[i] == 2 and seq[i + 1] == 1:
            range_1_2 += 1

    return range_1, range_2, range_1_2


# ==========================================
# 第二部分：KMeans.py 核心数据处理逻辑
# ==========================================

def process_folder(folder_path, data_list, file_names):
    """
    递归读取文件夹，收集 DataFrame 和对应的文件名
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            process_folder(item_path, data_list, file_names)
        elif item.endswith('.csv'):
            file_path = item_path
            try:
                df = pd.read_csv(file_path)
                data_list.append(df)
                file_names.append(item)
            except Exception as e:
                print(f"Error processing file: {file_path}, Error: {e}")


def extract_features(group):
    """
    提取基础时序特征
    已移除: Dst_One_Exists, Type_Unique_Count
    """
    features = {}

    # 第一阶段的时间阈值设定为前1秒
    start_time = group['Time'].iloc[0]
    phase_one_threshold = start_time + 1

    # 第一阶段中第一个报文与最后一个报文的时间差
    first_stage_group = group[group['Time'] <= phase_one_threshold]
    if not first_stage_group.empty:
        first_stage_time_diff = first_stage_group['Time'].iloc[-1] - first_stage_group['Time'].iloc[0]
        next_packet_after_first_stage = group[group['Time'] > phase_one_threshold]['Time'].min() if (
                group['Time'] > phase_one_threshold).any() else 0
    else:
        first_stage_time_diff = 0
        next_packet_after_first_stage = 0

    # 提取Time在next_packet_after_first_stage到next_packet_after_first_stage+1之间的Type序列
    if next_packet_after_first_stage > 0:
        start_time_range = next_packet_after_first_stage
        end_time_range = next_packet_after_first_stage + 1
        time_range_group = group[(group['Time'] >= start_time_range) & (group['Time'] < end_time_range)]
        if not time_range_group.empty:
            type_sequence_in_range = ','.join(map(str, time_range_group['Type'].tolist()))
        else:
            type_sequence_in_range = ''
    else:
        type_sequence_in_range = ''
    features['Type_Sequence_In_Range'] = type_sequence_in_range

    features['First_Stage_Time_Diff'] = first_stage_time_diff
    features['Next_Packet_After_First_Stage'] = next_packet_after_first_stage

    # 计算Dst为1时对应的平均Time
    avg_time_dst_1 = group[group['Dst'] == 1]['Time'].mean() if (group['Dst'] == 1).any() else float('nan')
    features['Avg_Time_Dst_1'] = avg_time_dst_1

    # Time大于1时Dst为1的数量
    dst_one_count = (group[group['Time'] > 1]['Dst'] == 1).sum()
    features['Dst_One_Count'] = dst_one_count

    # Time小于1部分的Type序列
    time_less_than_1_group = group[group['Time'] < 1]
    if not time_less_than_1_group.empty:
        type_sequence = ','.join(map(str, time_less_than_1_group['Type'].tolist()))
    else:
        type_sequence = ''
    features['Type_Sequence'] = type_sequence

    # 在Time小于1的情况下，Type中1-6分别第一次出现的Time
    for t in range(1, 7):
        first_occurrence_time = time_less_than_1_group[time_less_than_1_group['Type'] == t]['Time'].min() if (
                time_less_than_1_group['Type'] == t).any() else 0
        features[f'Type_{t}_First_Occurrence_Time'] = first_occurrence_time

    # 在Time小于1的情况下，Type中1-6分别出现的次数
    for t in range(1, 7):
        type_count = (time_less_than_1_group['Type'] == t).sum()
        features[f'Type_{t}_Count'] = type_count

    return pd.Series(features, name=group.name)


def extract_label_from_filename(filename):
    """
    从文件名提取 Label: 第二个 '-' 符号前的内容
    例: 'HONOR-60SE-2.csv' -> 'HONOR-60SE'
    例: 'MI-11-Pro-1.csv' -> 'MI-11-Pro'
    例: 'ABC-DEF.csv' -> 'ABC-DEF' (如果只有两个部分，取全部)
    例: 'ABC.csv' -> 'ABC' (如果没有横杠，取全部)
    """
    basename = os.path.basename(filename)
    # 去除后缀
    name_without_ext = os.path.splitext(basename)[0]

    parts = name_without_ext.split('-')

    if len(parts) >= 3:
        # 如果有至少3部分 (即至少2个横杠)，取前两部分拼接
        return '-'.join(parts[:2])
    else:
        # 否则返回整个文件名（不含后缀）
        return name_without_ext


# ==========================================
# 第三部分：主执行流程
# ==========================================

def main():
    print("🚀 开始处理数据...")

    # 1. 读取所有原始数据并记录文件名
    data_list = []
    file_names = []

    if not os.path.exists(BASE_PATH):
        print(f"❌ 错误：找不到基础路径 {BASE_PATH}，请检查路径配置。")
        return

    process_folder(BASE_PATH, data_list, file_names)

    if len(data_list) == 0:
        print("❌ 错误：未找到任何 CSV 文件。")
        return

    print(f"✅ 已加载 {len(data_list)} 个文件。")

    # 2. 合并数据并添加 Sample ID
    combined_data = pd.concat(data_list, keys=range(len(data_list)), names=['Sample', 'Row']).reset_index(
        level='Sample')

    # 将文件名映射到 Sample ID
    sample_to_file = dict(zip(range(len(file_names)), file_names))
    combined_data['FileName'] = combined_data['Sample'].map(sample_to_file)

    # 3. 提取 KMeans.py 定义的基础特征
    print("⏳ 正在提取基础时序特征...")
    extracted_features = combined_data.groupby('Sample').apply(extract_features).fillna(0)

    # 过滤掉无法提取特征的样本
    valid_indices = ~extracted_features.isnull().all(axis=1)
    new_features = extracted_features[valid_indices].reset_index(drop=True)

    # 提取对应的文件名和 Label
    valid_sample_ids = valid_indices[valid_indices].index.tolist()
    valid_files = [sample_to_file[sid] for sid in valid_sample_ids]
    valid_labels = [extract_label_from_filename(f) for f in valid_files]

    # 添加 Label 列
    new_features['Label'] = valid_labels

    print(f"📊 基础特征提取完成，样本数: {len(new_features)}")
    print(f"   Label 分布示例: {new_features['Label'].value_counts().head()}")

    # 4. 数据过滤 (基于 Next_Packet_After_First_Stage 的异常值处理)
    print("⏳ 正在进行异常值过滤...")
    label_stats = new_features.groupby('Label')['Next_Packet_After_First_Stage'].agg(['mean', 'std']).reset_index()
    threshold_factor = 3

    filtered_indices = []
    for index, row in new_features.iterrows():
        label = row['Label']
        if label not in label_stats['Label'].values:
            continue
        mean_val = label_stats[label_stats['Label'] == label]['mean'].values[0]
        std_val = label_stats[label_stats['Label'] == label]['std'].values[0]
        value = row['Next_Packet_After_First_Stage']

        if pd.isna(std_val) or std_val == 0:
            if abs(value - mean_val) == 0:
                filtered_indices.append(index)
        else:
            if abs(value - mean_val) <= threshold_factor * std_val:
                filtered_indices.append(index)

    filtered_new_features = new_features.loc[filtered_indices].reset_index(drop=True)
    print(f"✅ 过滤后剩余样本数: {len(filtered_new_features)}")

    # 5. 嵌入 sequence_feature.py 的序列特征提取
    print("⏳ 正在提取深度序列特征 (Excess, Lack, Range, lk_elem)...")

    # [已移除] 6_Count 的计算，避免与 Type_6_Count 重复

    # 5.1 计算 excess 和 lack
    filtered_new_features[['excess', 'lack']] = filtered_new_features['Type_Sequence'].apply(
        lambda x: pd.Series(calculate_excess_lack(x))
    )

    # 5.2 计算 Range 相关特征
    filtered_new_features[['Range_1', 'Range_2', 'Range_1_2']] = filtered_new_features['Type_Sequence_In_Range'].apply(
        lambda x: pd.Series(calculate_range_features(x))
    )

    # 5.3 计算 lk_count (即 Lack_Count)
    filtered_new_features['lk_count'] = filtered_new_features['lack'].apply(
        lambda x: len(x) if isinstance(x, tuple) else 0)

    # 5.4 强制生成 lk_elem_1 到 lk_elem_5
    lk_elem_cols = []
    for i in range(1, 6):  # 1 到 5
        col_name = f'lk_elem_{i}'
        # 如果元组长度不够，填充 0
        filtered_new_features[col_name] = filtered_new_features['lack'].apply(
            lambda x: x[i - 1] if isinstance(x, tuple) and len(x) >= i else 0
        )
        lk_elem_cols.append(col_name)

    print(f"   已生成固定列: {', '.join(lk_elem_cols)}")

    # 6. 整理列顺序并保存
    print(f"💾 正在保存至 {OUTPUT_FILE} ...")

    # 定义必须在前面的三列
    priority_cols = ['Label', 'Type_Sequence', 'Type_Sequence_In_Range']

    # 定义新生成的序列特征列 (除了上面三个)
    # 注意：这里不再包含 '6_Count'
    new_seq_cols = ['excess', 'lk_count'] + lk_elem_cols + ['Range_1', 'Range_2', 'Range_1_2']

    # 获取所有其他列 (基础时序特征，如 Type_1_Count, Type_6_Count 等)
    all_cols = list(filtered_new_features.columns)
    other_cols = [c for c in all_cols if c not in priority_cols and c not in new_seq_cols]

    # 最终列顺序: Priority + Others + New_Seq
    final_columns = priority_cols + other_cols + new_seq_cols

    # 确保所有列都在 DataFrame 中
    final_columns = [c for c in final_columns if c in filtered_new_features.columns]

    # 保存
    filtered_new_features[final_columns].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"✅ 成功！所有特征已保存至: {OUTPUT_FILE}")
    print(f"📊 最终数据维度: {filtered_new_features.shape}")
    print(f"📋 前 15 列预览: {final_columns[:15]}")

    # 预览
    print("\n🔍 数据预览 (前 2 行):")
    print(filtered_new_features[final_columns].head(2))


if __name__ == "__main__":
    main()