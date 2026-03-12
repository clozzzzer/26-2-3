import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, GlobalAveragePooling1D, Dropout, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =================配置区域=================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
# =========================================

# 1. 数据读取与预处理
file_path = 'new_feature_csv.csv'
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("文件读取成功！")
except Exception as e:
    print(f"读取失败: {e}")
    raise e

df.columns = df.columns.str.strip()


# --- 解析函数 ---
def parse_to_ids(x):
    """将字符串列表/元组解析为整数ID列表"""
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x in ['()', '[]', '']: return []
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")): x = x[1:-1]
    try:
        res = ast.literal_eval(x)
        if isinstance(res, tuple): res = list(res)
        if not isinstance(res, list): return []
        # 确保所有元素是整数
        return [int(i) for i in res if str(i).isdigit()]
    except:
        return []


# 应用解析
df['excess_ids'] = df['excess'].apply(parse_to_ids)
df['lack_ids'] = df['lack'].apply(parse_to_ids)

# 2. 数据准备
# --- A. 数值特征 ---
numeric_cols = ['6_Count', 'Range_1', 'Range_2', 'Range_1_2', 'Lack_Count']
# 过滤掉不存在的列
available_numeric_cols = [c for c in numeric_cols if c in df.columns]
X_numeric_raw = df[available_numeric_cols].fillna(0).values

# --- B. 序列特征 (Excess & Lack) ---
# 为了输入神经网络，我们需要将变长序列填充到最大长度
max_len_excess = max(len(x) for x in df['excess_ids'])
max_len_lack = max(len(x) for x in df['lack_ids'])

# 设置一个合理的最大长度限制 (防止个别异常值导致矩阵过大)，例如 20
MAX_LEN = max(20, max_len_excess, max_len_lack)


def pad_sequences(sequences, maxlen, value=0):
    """手动填充序列"""
    result = []
    for seq in sequences:
        if len(seq) > maxlen:
            result.append(seq[:maxlen])
        else:
            result.append(seq + [value] * (maxlen - len(seq)))
    return np.array(result)


X_excess_seq = pad_sequences(df['excess_ids'], MAX_LEN)
X_lack_seq = pad_sequences(df['lack_ids'], MAX_LEN)

# --- C. 标签编码 ---
if 'Label' not in df.columns:
    raise ValueError("缺少 'Label' 列")

le = LabelEncoder()
y_labels = le.fit_transform(df['Label'])
num_classes = len(le.classes_)
y_cat = to_categorical(y_labels, num_classes)

print(f"数据形状:")
print(f"  - 数值特征: {X_numeric_raw.shape}")
print(f"  - Excess 序列: {X_excess_seq.shape}")
print(f"  - Lack 序列: {X_lack_seq.shape}")
print(f"  - 类别数量: {num_classes}")

# 3. 划分数据集
# 注意：对于深度学习，通常需要更多的数据。这里保持 0.3 测试集
X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_numeric_raw, y_cat, test_size=0.3,
                                                                    random_state=42, stratify=y_labels)
X_ex_train, X_ex_test = train_test_split(X_excess_seq, test_size=0.3, random_state=42,
                                         shuffle=False)  # shuffle=False 保持对应关系
X_la_train, X_la_test = train_test_split(X_lack_seq, test_size=0.3, random_state=42, shuffle=False)

# 重新打乱 y_num_test 对应的索引逻辑 (train_test_split 默认shuffle=True，所以上面单独split序列会导致错位)
# 修正：必须一起 split
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y_labels)

X_num_train = X_numeric_raw[train_idx]
X_num_test = X_numeric_raw[test_idx]

X_ex_train = X_excess_seq[train_idx]
X_ex_test = X_excess_seq[test_idx]

X_la_train = X_lack_seq[train_idx]
X_la_test = X_lack_seq[test_idx]

y_train = y_cat[train_idx]
y_test = y_cat[test_idx]
y_test_labels = y_labels[test_idx]  # 用于计算混淆矩阵的原始标签

# 标准化数值特征 (深度学习对数值敏感)
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_test = scaler.transform(X_num_test)


# 4. 构建混合深度学习模型
def build_hybrid_model(num_numeric_features, max_len, num_classes, vocab_size_ex=50, vocab_size_la=50):
    # --- 输入层 ---
    input_numeric = Input(shape=(num_numeric_features,), name='numeric_input')
    input_excess = Input(shape=(max_len,), name='excess_input')
    input_lack = Input(shape=(max_len,), name='lack_input')

    # --- 数值分支 ---
    x_num = Dense(64, activation='relu')(input_numeric)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.3)(x_num)

    # --- Excess 分支 (Embedding + Pooling) ---
    # Embedding: 将 ID 映射为向量。vocab_size 设为最大ID+1，或者一个足够大的数
    # output_dim: 嵌入维度，设为 8 或 16
    emb_ex = Embedding(input_dim=vocab_size_ex, output_dim=16, mask_zero=True)(input_excess)
    # GlobalAveragePooling: 将变长序列平均为一个固定向量，忽略 padding 的 0
    pool_ex = GlobalAveragePooling1D()(emb_ex)
    x_ex = Dense(32, activation='relu')(pool_ex)
    x_ex = Dropout(0.3)(x_ex)

    # --- Lack 分支 (Embedding + Pooling) ---
    emb_la = Embedding(input_dim=vocab_size_la, output_dim=16, mask_zero=True)(input_lack)
    pool_la = GlobalAveragePooling1D()(emb_la)
    x_la = Dense(32, activation='relu')(pool_la)
    x_la = Dropout(0.3)(x_la)

    # --- 融合层 ---
    combined = Concatenate()([x_num, x_ex, x_la])
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # --- 输出层 ---
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_numeric, input_excess, input_lack], outputs=output)
    return model


# 确定 Vocabulary Size (最大 ID + 1)
# 防止 ID 过大导致 Embedding 层过大，可以截断或映射
max_id_ex = np.max(X_excess_seq) + 1
max_id_la = np.max(X_lack_seq) + 1
vocab_ex = max(50, max_id_ex)  # 至少 50
vocab_la = max(50, max_id_la)

print(f"\n构建模型... (Excess Vocab: {vocab_ex}, Lack Vocab: {vocab_la})")
model = build_hybrid_model(
    num_numeric_features=X_num_train.shape[1],
    max_len=MAX_LEN,
    num_classes=num_classes,
    vocab_size_ex=vocab_ex,
    vocab_size_la=vocab_la
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5. 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n开始训练...")
history = model.fit(
    {'numeric_input': X_num_train, 'excess_input': X_ex_train, 'lack_input': X_la_train},
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 6. 评估与预测
print("\n--- 模型评估 ---")
y_pred_proba = model.predict({'numeric_input': X_num_test, 'excess_input': X_ex_test, 'lack_input': X_la_test})
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred)
labels_unique = sorted(df['Label'].unique())
# 将预测的索引转回原始标签
y_pred_labels = le.inverse_transform(y_pred)

print(f"测试集准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test_labels, y_pred_labels, digits=4))

# 绘制混淆矩阵
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels_unique)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_unique,
            yticklabels=labels_unique)
plt.title(f'Deep Learning Hybrid Model\nAccuracy: {accuracy:.2%}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

print("\n程序执行完毕。")