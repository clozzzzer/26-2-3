import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ==========================================
# 配置区域
# ==========================================
BASE_PATH = "./data/csv0419_1"
OUTPUT_DIR = "./data/feature/csv0419_1"
OUTPUT_FILE_ALL = os.path.join(OUTPUT_DIR, "All_feature_csv0419_1.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 第一部分：序列特征计算函数 (保持不变)
# ==========================================

def calculate_excess_lack(seq_str):
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


def calculate_binary_lack_flags(seq_str):
    flags = {}
    try:
        if pd.isna(seq_str) or str(seq_str).strip() == '':
            for i in range(1, 6):
                flags[f'lk_elem_{i}'] = 1
            return flags

        seq = [int(x.strip()) for x in str(seq_str).split(',') if x.strip()]
        valid_seq = [num for num in seq if num != 6]
        unique_types = set(valid_seq)

        for i in range(1, 6):
            if i not in unique_types:
                flags[f'lk_elem_{i}'] = 1
            else:
                flags[f'lk_elem_{i}'] = 0
    except Exception:
        for i in range(1, 6):
            flags[f'lk_elem_{i}'] = 0

    return flags


# ==========================================
# 第二部分：核心数据处理逻辑
# ==========================================

def process_folder(folder_path, data_list, file_names):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            process_folder(item_path, data_list, file_names)
        elif item.endswith('.csv'):
            try:
                df = pd.read_csv(item_path)
                data_list.append(df)
                file_names.append(item)
            except Exception as e:
                print(f"Error reading {item_path}: {e}")


def extract_features(group):
    features = {}

    # --- 1. 定义阶段 A (Phase A) ---
    type_6_indices = group[group['Type'] == 6].index.tolist()

    if type_6_indices:
        last_6_idx = type_6_indices[-1]
        phase_a_group = group.loc[:last_6_idx].copy()
        type_sequence = ','.join(map(str, phase_a_group['Type'].tolist()))
    else:
        phase_a_group = group.iloc[0:0]
        type_sequence = ''

    features['Type_Sequence'] = type_sequence

    # --- 2. 时间相关特征 (全局) ---
    start_time = group['Time'].iloc[0]
    phase_one_threshold = start_time + 1

    first_stage_group = group[group['Time'] <= phase_one_threshold]
    if not first_stage_group.empty:
        first_stage_time_diff = first_stage_group['Time'].iloc[-1] - first_stage_group['Time'].iloc[0]
        next_packet_after_first_stage = group[group['Time'] > phase_one_threshold]['Time'].min() if (
                group['Time'] > phase_one_threshold).any() else 0
    else:
        first_stage_time_diff = 0
        next_packet_after_first_stage = 0

    features['First_Stage_Time_Diff'] = first_stage_time_diff
    features['Next_Packet_After_First_Stage'] = next_packet_after_first_stage

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

    # --- 3. Dst 特征 (基于阶段 A) ---
    if not phase_a_group.empty:
        has_dst_one = (phase_a_group['Dst'] == 1).any()
        features['Dst_One_Exist'] = 1 if has_dst_one else 0
        features['Dst_One_Count'] = (phase_a_group['Dst'] == 1).sum()
    else:
        features['Dst_One_Exist'] = 0
        features['Dst_One_Count'] = 0

    # --- 4. Type 统计与 Last_Time (基于阶段 A) ---
    for t in range(1, 7):
        features[f'Type_{t}_First_Occurrence_Time'] = 0
        features[f'Type_{t}_Last_Time'] = 0
        features[f'Type_{t}_Count'] = 0

    if not phase_a_group.empty and type_sequence:
        for t in range(1, 7):
            subset = phase_a_group[phase_a_group['Type'] == t]
            if not subset.empty:
                features[f'Type_{t}_First_Occurrence_Time'] = subset['Time'].min()
                features[f'Type_{t}_Last_Time'] = subset['Time'].max()
                features[f'Type_{t}_Count'] = len(subset)

    # --- 5. 时间差特征计算 ---
    t1_first = features['Type_1_First_Occurrence_Time']
    t2_last = features['Type_2_Last_Time']
    t3_first = features['Type_3_First_Occurrence_Time']
    t3_last = features['Type_3_Last_Time']
    t4_first = features['Type_4_First_Occurrence_Time']
    t4_last = features['Type_4_Last_Time']
    t5_first = features['Type_5_First_Occurrence_Time']
    t5_last = features['Type_5_Last_Time']
    t6_first = features['Type_6_First_Occurrence_Time']
    t6_last = features['Type_6_Last_Time']

    def safe_diff(end_t, start_t):
        if end_t > 0 and start_t > 0:
            return max(0.0, end_t - start_t)
        return 0.0

    features['Probe_Time'] = safe_diff(t2_last, t1_first)
    features['Assoc_Time'] = safe_diff(t5_last, t4_first)
    features['Gap_23_Time'] = safe_diff(t3_first, t2_last)
    features['Gap_34_Time'] = safe_diff(t4_first, t3_last)
    features['Gap_45_Time'] = safe_diff(t5_first, t4_last)
    features['Type_6_Time'] = safe_diff(t6_last, t6_first)

    # --- 6. 计算时间差占比 (Ratio) ---
    total_duration = features['First_Stage_Time_Diff']
    time_diff_cols = ['Probe_Time', 'Assoc_Time', 'Gap_23_Time', 'Gap_34_Time', 'Gap_45_Time', 'Type_6_Time',
                      'First_Stage_Time_Diff']

    for col in time_diff_cols:
        val = features[col]
        if total_duration > 0:
            features[f'{col}_Ratio'] = val / total_duration
        else:
            features[f'{col}_Ratio'] = 0.0

    # --- 7. 高级统计特征 (基于阶段 A) ---
    if len(phase_a_group) > 1:
        # 计算帧间间隔 (IAT)
        iat = phase_a_group['Time'].diff().dropna()

        # 基础统计量
        features['A_Mean_IAT'] = iat.mean()
        features['A_IAT_StdDev'] = iat.std()

        # 🆕 分位数特征 (Quantile Features)
        if len(iat) > 0:
            features['A_IAT_Q1'] = iat.quantile(0.25)
            features['A_IAT_Median'] = iat.quantile(0.50)
            features['A_IAT_Q3'] = iat.quantile(0.75)
            features['A_IAT_IQR'] = features['A_IAT_Q3'] - features['A_IAT_Q1']
            features['A_IAT_P10'] = iat.quantile(0.10)
            features['A_IAT_P90'] = iat.quantile(0.90)
            features['A_IAT_P95'] = iat.quantile(0.95)

            if features['A_IAT_Median'] > 0:
                features['A_IAT_Skew_Ratio'] = (features['A_IAT_P90'] - features['A_IAT_Median']) / features[
                    'A_IAT_Median']
            else:
                features['A_IAT_Skew_Ratio'] = 0.0
        else:
            for key in ['A_IAT_Q1', 'A_IAT_Median', 'A_IAT_Q3', 'A_IAT_IQR', 'A_IAT_P10', 'A_IAT_P90', 'A_IAT_P95',
                        'A_IAT_Skew_Ratio']:
                features[key] = 0.0

        # A_CV
        if features['A_Mean_IAT'] > 0:
            features['A_CV'] = features['A_IAT_StdDev'] / features['A_Mean_IAT']
        else:
            features['A_CV'] = 0.0

        # A_Skewness, A_Kurtosis
        if len(iat) > 2:
            features['A_Skewness'] = skew(iat)
            features['A_Kurtosis'] = kurtosis(iat)
        else:
            features['A_Skewness'] = 0.0
            features['A_Kurtosis'] = 0.0

        # A_Retran_intervals
        retrans_intervals = []
        sorted_phase_a = phase_a_group.sort_index()
        types = sorted_phase_a['Type'].values
        times = sorted_phase_a['Time'].values

        for i in range(1, len(types)):
            if types[i] == types[i - 1]:
                diff = times[i] - times[i - 1]
                if diff >= 0:
                    retrans_intervals.append(diff)

        if len(retrans_intervals) > 0:
            features['A_Retran_intervals'] = np.mean(retrans_intervals)
        else:
            features['A_Retran_intervals'] = 0.0

        # ---------------------------------------------------------
        # 🚀 优化：Burst 检测 (基于 Median)
        # ---------------------------------------------------------
        burst_threshold = features['A_IAT_Median']
        if burst_threshold <= 0:
            if features['A_Mean_IAT'] > 0:
                burst_threshold = features['A_Mean_IAT'] * 0.5
            else:
                burst_threshold = 0

        if burst_threshold > 0:
            is_burst = iat < burst_threshold
            bursts = []
            current_len = 0
            for val in is_burst:
                if val:
                    current_len += 1
                else:
                    if current_len > 0:
                        bursts.append(current_len + 1)
                        current_len = 0
            if current_len > 0:
                bursts.append(current_len + 1)

            total_burst_packets = sum(bursts)
            total_packets = len(phase_a_group)

            features['A_Burst_Rate'] = total_burst_packets / total_packets if total_packets > 0 else 0.0
            features['A_Burst_Duration'] = iat[is_burst].sum()
        else:
            features['A_Burst_Rate'] = 0.0
            features['A_Burst_Duration'] = 0.0

        # ---------------------------------------------------------
        # 🆕 新增：静默检测 (Silence Detection)
        # ---------------------------------------------------------
        silence_threshold = features['A_IAT_P95']
        if silence_threshold <= 0:
            silence_threshold = features['A_Mean_IAT'] * 2.0

        if silence_threshold > 0:
            is_silence = iat > silence_threshold
            silence_iats = iat[is_silence]

            features['A_Silence_Count'] = int(is_silence.sum())
            features['A_Silence_Rate'] = features['A_Silence_Count'] / len(iat) if len(iat) > 0 else 0.0

            if len(silence_iats) > 0:
                features['A_Silence_Total_Duration'] = silence_iats.sum()
                features['A_Silence_Max_Duration'] = silence_iats.max()
                features['A_Silence_Avg_Duration'] = silence_iats.mean()
            else:
                features['A_Silence_Total_Duration'] = 0.0
                features['A_Silence_Max_Duration'] = 0.0
                features['A_Silence_Avg_Duration'] = 0.0
        else:
            features['A_Silence_Count'] = 0
            features['A_Silence_Rate'] = 0.0
            features['A_Silence_Total_Duration'] = 0.0
            features['A_Silence_Max_Duration'] = 0.0
            features['A_Silence_Avg_Duration'] = 0.0

        # ---------------------------------------------------------
        # 🆕 新增：马尔科夫转移矩阵展平 (Markov Transition Matrix Flattened)
        # ---------------------------------------------------------
        trans_matrix = np.zeros((6, 6), dtype=int)
        type_list = types.tolist() if not isinstance(types, list) else types

        for i in range(len(type_list) - 1):
            curr_t = type_list[i]
            next_t = type_list[i + 1]
            if 1 <= curr_t <= 6 and 1 <= next_t <= 6:
                row_idx = curr_t - 1
                col_idx = next_t - 1
                trans_matrix[row_idx, col_idx] += 1

        for r in range(6):
            for c in range(6):
                from_type = r + 1
                to_type = c + 1
                features[f'Trans_{from_type}_{to_type}'] = int(trans_matrix[r, c])

    else:
        # 样本数 <= 1 的默认值处理
        features['A_Mean_IAT'] = 0.0
        features['A_IAT_StdDev'] = 0.0
        features['A_Retran_intervals'] = 0.0
        features['A_Burst_Rate'] = 0.0
        features['A_Burst_Duration'] = 0.0
        features['A_CV'] = 0.0
        features['A_Skewness'] = 0.0
        features['A_Kurtosis'] = 0.0
        for key in ['A_IAT_Q1', 'A_IAT_Median', 'A_IAT_Q3', 'A_IAT_IQR', 'A_IAT_P10', 'A_IAT_P90', 'A_IAT_P95',
                    'A_IAT_Skew_Ratio']:
            features[key] = 0.0
        features['A_Silence_Count'] = 0
        features['A_Silence_Rate'] = 0.0
        features['A_Silence_Total_Duration'] = 0.0
        features['A_Silence_Max_Duration'] = 0.0
        features['A_Silence_Avg_Duration'] = 0.0
        for r in range(1, 7):
            for c in range(1, 7):
                features[f'Trans_{r}_{c}'] = 0

    return pd.Series(features, name=group.name)


def extract_label_from_filename(filename):
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:2])
    else:
        return name_without_ext


# ==========================================
# 第三部分：基础清洗
# ==========================================
def basic_clean_data(df):
    initial_count = len(df)
    print(f"🧹 开始基础清洗... 初始样本数: {initial_count}")

    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        valid_labels = label_counts[label_counts > 1].index.tolist()
        removed_by_label = initial_count - len(df[df['Label'].isin(valid_labels)])
        df = df[df['Label'].isin(valid_labels)].reset_index(drop=True)
        print(f"   - 规则1 (类别样本数<=1): 移除了 {removed_by_label} 个样本。")

    count_cols = [f'Type_{i}_Count' for i in range(1, 7)]
    existing_count_cols = [col for col in count_cols if col in df.columns]

    if existing_count_cols:
        mask_excess = (df[existing_count_cols] > 10).any(axis=1)
        removed_by_count = mask_excess.sum()
        df = df[~mask_excess].reset_index(drop=True)
        print(f"   - 规则2 (Type_Count > 10): 移除了 {removed_by_count} 个样本。")

    print(f"✅ 基础清洗完成。剩余样本数: {len(df)}")
    return df


# ==========================================
# 第四部分：高级分类与清洗
# ==========================================

def advanced_classification_and_cleaning(df):
    print("\n🚀 开始高级分类与清洗...")

    print("   - 进行第一级分类 (D1/D0)...")

    def classify_level1(group):
        total = len(group)
        if total == 0: return 'D0'
        count_1 = (group['Dst_One_Exist'] == 1).sum()
        return 'D1' if count_1 / total > 0.5 else 'D0'

    label_class_map = df.groupby('Label').apply(classify_level1)
    df['Class_Level1'] = df['Label'].map(label_class_map)
    print(f"      第一级分类分布:\n{df['Class_Level1'].value_counts()}")

    print("   - 进行第一级清洗...")
    mask_d1_clean = (df['Class_Level1'] == 'D1') & (df['Dst_One_Exist'] == 0)
    mask_d0_clean = (df['Class_Level1'] == 'D0') & (df['Dst_One_Exist'] == 1)
    remove_mask_l1 = mask_d1_clean | mask_d0_clean
    removed_l1 = remove_mask_l1.sum()
    df = df[~remove_mask_l1].reset_index(drop=True)
    print(f"      第一级清洗移除了 {removed_l1} 个不纯样本。剩余: {len(df)}")

    if len(df) == 0: return None

    print("   - 进行第二级分类 (NOPR/PR)...")

    def classify_level2(group):
        total = len(group)
        if total == 0: return 'PR'
        mask_nopr = (group['Type_1_Count'] == 0) & (group['Type_2_Count'] == 0)
        count_nopr = mask_nopr.sum()
        return 'NOPR' if count_nopr / total > 0.5 else 'PR'

    df_d1 = df[df['Class_Level1'] == 'D1'].copy()
    df_d0 = df[df['Class_Level1'] == 'D0'].copy()

    class_l2_d1 = df_d1.groupby('Label').apply(classify_level2) if not df_d1.empty else pd.Series(dtype=str)
    class_l2_d0 = df_d0.groupby('Label').apply(classify_level2) if not df_d0.empty else pd.Series(dtype=str)

    df['Class_Level2'] = ''
    if not df_d1.empty:
        df.loc[df['Class_Level1'] == 'D1', 'Class_Level2'] = df.loc[df['Class_Level1'] == 'D1', 'Label'].map(
            class_l2_d1)
    if not df_d0.empty:
        df.loc[df['Class_Level1'] == 'D0', 'Class_Level2'] = df.loc[df['Class_Level1'] == 'D0', 'Label'].map(
            class_l2_d0)

    df['Final_Class'] = df['Class_Level1'] + df['Class_Level2']
    print(f"      最终四类分布:\n{df['Final_Class'].value_counts()}")

    print("   - 进行第二级清洗...")
    remove_mask_l2 = pd.Series([False] * len(df))

    mask_nopr = df['Final_Class'].isin(['D1NOPR', 'D0NOPR'])
    mask_nopr_impure = (df['Type_1_Count'] != 0) | (df['Type_2_Count'] != 0)
    remove_mask_l2 |= mask_nopr & mask_nopr_impure

    mask_pr = df['Final_Class'].isin(['D1PR', 'D0PR'])
    mask_pr_impure = (df['Type_1_Count'] == 0) & (df['Type_2_Count'] == 0)
    remove_mask_l2 |= mask_pr & mask_pr_impure

    removed_l2 = remove_mask_l2.sum()
    df = df[~remove_mask_l2].reset_index(drop=True)
    print(f"      第二级清洗移除了 {removed_l2} 个不纯样本。剩余: {len(df)}")

    return df


# ==========================================
# 第五部分：主执行流程
# ==========================================

def main():
    print("🚀 开始处理数据 (版本 v13: Label列移至第一列)...")

    data_list = []
    file_names = []

    if os.path.exists(BASE_PATH):
        process_folder(BASE_PATH, data_list, file_names)
    else:
        print(f"❌ 未找到路径 {BASE_PATH}。")
        return

    if len(data_list) == 0:
        print("❌ 没有数据可处理。")
        return

    print(f"✅ 已加载 {len(data_list)} 个文件。")

    combined_data = pd.concat(data_list, keys=range(len(data_list)), names=['Sample', 'Row']).reset_index(
        level='Sample')
    sample_to_file = dict(zip(range(len(file_names)), file_names))
    combined_data['FileName'] = combined_data['Sample'].map(sample_to_file)

    print("⏳ 正在提取基础特征...")
    extracted_features = combined_data.groupby('Sample').apply(extract_features)

    if isinstance(extracted_features, pd.Series):
        extracted_features = extracted_features.unstack()

    if not isinstance(extracted_features, pd.DataFrame):
        records = []
        for idx, row in extracted_features.items():
            if isinstance(row, pd.Series):
                records.append(row.to_dict())
        if records:
            extracted_features = pd.DataFrame(records)
        else:
            print("❌ 无法重构 DataFrame。")
            return

    valid_indices = ~extracted_features.isnull().all(axis=1)
    new_features = extracted_features[valid_indices].reset_index(drop=True)

    if len(new_features) == 0:
        print("❌ 过滤后没有剩余数据。")
        return

    valid_sample_ids = extracted_features.index[valid_indices].tolist()
    valid_files = [sample_to_file[sid] for sid in valid_sample_ids if sid in sample_to_file]
    valid_labels = [extract_label_from_filename(f) for f in valid_files]

    if len(valid_labels) != len(new_features):
        min_len = min(len(valid_labels), len(new_features))
        valid_labels = valid_labels[:min_len]
        new_features = new_features.iloc[:min_len]

    new_features['Label'] = valid_labels
    print(f"📊 基础特征提取完成，有效样本数: {len(new_features)}")

    print("⏳ 计算深度序列特征...")

    new_features[['excess', 'lack']] = new_features['Type_Sequence'].apply(
        lambda x: pd.Series(calculate_excess_lack(x))
    )
    new_features[['Range_1', 'Range_2', 'Range_1_2']] = new_features['Type_Sequence_In_Range'].apply(
        lambda x: pd.Series(calculate_range_features(x))
    )
    new_features['lk_count'] = new_features['lack'].apply(lambda x: len(x) if isinstance(x, tuple) else 0)

    binary_flags_series = new_features['Type_Sequence'].apply(calculate_binary_lack_flags)
    binary_flags_df = pd.DataFrame.from_records(binary_flags_series.tolist(), index=new_features.index)
    expected_cols = [f'lk_elem_{i}' for i in range(1, 6)]
    for col in expected_cols:
        if col not in binary_flags_df.columns:
            binary_flags_df[col] = 0
    binary_flags_df = binary_flags_df[expected_cols]
    for col in binary_flags_df.columns:
        new_features[col] = binary_flags_df[col]

    # 🔄 【关键修改】在保存前，将 Label 列移动到第一列
    # 逻辑：获取所有列名 -> 移除 'Label' -> 将 'Label' 插入到列表头部 -> 重排 DataFrame
    if 'Label' in new_features.columns:
        cols = new_features.columns.tolist()
        cols.remove('Label')
        cols = ['Label'] + cols
        new_features = new_features[cols]
        print("✅ 已将 'Label' 列调整至第一列。")

    print(f"💾 保存清洗前完整数据至: {OUTPUT_FILE_ALL}")
    new_features.to_csv(OUTPUT_FILE_ALL, index=False, encoding='utf-8-sig')
    print(f"   ✅ 已保存 {len(new_features)} 条记录。")

    new_features = basic_clean_data(new_features)
    if len(new_features) == 0:
        print("❌ 基础清洗后无数据。")
        return

    final_df = advanced_classification_and_cleaning(new_features)

    if final_df is None or len(final_df) == 0:
        print("❌ 高级分类清洗后无数据。")
        return

    # 🔄 【关键修改】同样对清洗后的最终数据执行列重排，确保分类文件也是 Label 在第一列
    if 'Label' in final_df.columns:
        cols = final_df.columns.tolist()
        cols.remove('Label')
        cols = ['Label'] + cols
        final_df = final_df[cols]

    classes = ['D1NOPR', 'D1PR', 'D0NOPR', 'D0PR']
    print("\n💾 正在保存分类数据...")

    for cls_name in classes:
        sub_df = final_df[final_df['Final_Class'] == cls_name]
        filename = os.path.join(OUTPUT_DIR, f"{cls_name}_csv0419_1.csv")
        if len(sub_df) > 0:
            sub_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"   ✅ {cls_name}: {len(sub_df)} 条 -> {filename}")
        else:
            print(f"   ⚠️ {cls_name}: 无数据，跳过保存。")

    final_all_path = os.path.join(OUTPUT_DIR, "All_Cleaned_feature_csv0419_1.csv")
    final_df.to_csv(final_all_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 已保存清洗后的总文件: {final_all_path} ({len(final_df)} 条)")

    print("\n🎉 全部处理完成！")


if __name__ == "__main__":
    main()