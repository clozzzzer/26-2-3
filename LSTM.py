import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
# =========================================

# 1. 数据读取与预处理
# --- 文件路径 ---
# 请确保该文件与脚本在同一目录，或提供完整路径
#file_path = r'.\data/feature/csv0419_1/D0PR_csv0419_1_Cleaned_Final.csv'
file_path = r'.\data/feature/csv0419_1_2/D0PR_csv0419_1_2_Cleaned_Final.csv'
print(f"正在读取文件: {file_path} ...")

try:
    df = pd.read_csv(file_path, engine='python')
    print("✅ 文件读取成功！")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    raise e

# 去除列名两端的空格
df.columns = df.columns.str.strip()
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 检查 Label 列
if 'Label' not in df.columns:
    raise ValueError("❌ 数据集中缺少 'Label' 列。请检查列名是否为 'Label'。")

# 2. 特征与标签分离
# --- A. 特征 (X) ---
# 选择除了 'Label' 之外的所有列作为特征
# 假设所有非标签列都是数值特征
X = df.drop('Label', axis=1).values

# --- B. 标签 (y) ---
y_labels_raw = df['Label'].values

# 3. 标签编码与处理
print("\n--- 数据准备 ---")
le = LabelEncoder()
y_labels_encoded = le.fit_transform(y_labels_raw)
num_classes = len(le.classes_)
y_cat = to_categorical(y_labels_encoded, num_classes)

print(f"类别映射: {dict(enumerate(le.classes_))}")
print(f"特征矩阵形状: {X.shape}")
print(f"类别数量: {num_classes}")

# 4. 划分数据集与标准化
print("\n--- 划分与标准化 ---")
# 划分训练集和测试集 (70% 训练, 30% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.3, random_state=42, stratify=y_labels_encoded
)

# 标准化数值特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集形状: {X_train_scaled.shape}")
print(f"测试集形状: {X_test_scaled.shape}")


# 5. 构建深度学习模型 (纯数值版本)
def build_numeric_model(input_dim, num_classes):
    input_numeric = Input(shape=(input_dim,), name='numeric_input')

    # 隐藏层 1
    x = Dense(128, activation='relu')(input_numeric)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # 隐藏层 2
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # 隐藏层 3
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    # 输出层
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_numeric, outputs=output)
    return model


print("\n--- 模型构建 ---")
model = build_numeric_model(
    input_dim=X_train_scaled.shape[1],
    num_classes=num_classes
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 6. 训练模型
print("\n--- 开始训练 ---")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.5,  # 从训练集中再分出 20% 作为验证集
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 7. 评估与预测
print("\n--- 模型评估 ---")
# 在测试集上评估
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✅ 测试集准确率: {test_acc:.4f}")

# 预测
y_pred_proba = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred_proba, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 分类报告
print("\n📋 分类报告:")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

# 8. 可视化

# --- A. 混淆矩阵 ---
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'混淆矩阵\nAccuracy: {test_acc:.2%}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# --- B. 训练历史 ---
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

print("\n🎉 程序执行完毕。")