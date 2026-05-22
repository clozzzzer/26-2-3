import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import matplotlib

warnings.filterwarnings("ignore")


# ==========================================
# 🛠️ 核心修复：自动设置中文字体
# ==========================================
def setup_chinese_font():
    """
    自动检测并设置中文字体，解决中文乱码问题
    """
    # 常见的中文字体列表（按优先级排序）
    font_candidates = [
        'Microsoft YaHei',  # Windows 常用
        'SimHei',  # 黑体
        'PingFang SC',  # Mac 常用
        'Arial Unicode MS',  # Mac 旧版
        'WenQuanYi Micro Hei',  # Linux 常用
        'Noto Sans CJK SC'  # 通用
    ]

    found_font = False
    for font_name in font_candidates:
        try:
            # 尝试设置字体
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
            print(f"✅ 已加载字体: {font_name}")
            found_font = True
            break
        except:
            continue

    if not found_font:
        print("⚠️ 警告：未在系统中找到常见中文字体，图表中文可能无法正常显示。")
        print("   建议：请确保你的系统安装了 'Microsoft YaHei' 或 'SimHei'。")


# 在脚本开始时执行字体设置
setup_chinese_font()


# ==========================================
# 📊 绘图逻辑
# ==========================================

def letter_to_num(letter):
    """将Excel列字母转换为数字索引 (A=0, B=1, ...)"""
    letter = letter.upper()
    result = 0
    for char in letter:
        if 'A' <= char <= 'Z':
            result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


def plot_features_separate_windows(file_path, feature_input):
    """ 为每个特征生成一个独立的窗口，并支持中文显示 """

    # --- 2. 解析特征列 ---
    col_indices = []
    for letter in [x.strip() for x in feature_input.replace('，', ',').split(',') if x.strip()]:
        try:
            col_letter = ''.join(filter(str.isalpha, letter))
            col_indices.append(letter_to_num(col_letter))
        except:
            print(f"⚠️ 警告: 无法解析列名 '{letter}'")
    if not col_indices:
        print("❌ 错误: 未解析到有效列")
        return

    # --- 1. 数据读取 (位置提前) ---
    try:
        df_raw = pd.read_csv(file_path)
        print(f"✅ 成功读取文件: {file_path}")
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # --- 2.5 获取真实列名 ---
    col_names = [df_raw.columns[idx] for idx in col_indices]

    # --- 3. 准备配色方案 ---
    devices = df_raw['Label'].unique()  # 注意这里也改用 df_raw
    palette = sns.color_palette("husl", len(devices))
    color_dict = dict(zip(devices, palette))
    print(f"🚀 开始生成 {len(col_names)} 个特征的独立图表...")

    # --- 4. 循环绘制独立窗口 ---
    for feature_name in col_names:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_raw, x='Label', y=feature_name, palette=color_dict, width=0.5, fliersize=3)
        sns.stripplot(data=df_raw, x='Label', y=feature_name, palette=color_dict, alpha=0.6, size=5, jitter=True)

        plt.title(f'设备特征分布分析：{feature_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('设备名称 (Label)', fontsize=12)
        plt.ylabel('数值分布 (Value)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 自动调整布局
        plt.tight_layout()

        # 显示窗口
        # 显示窗口
        plt.show()
        print(f"   - 已生成图表: {feature_name}") # 修复：将 feature 改为 feature_name

    print("✅ 所有图表生成完毕。")


# ==========================================
# 🚀 运行配置
# ==========================================
if __name__ == "__main__":
    # --- 修改点 1: 输入文件路径 ---
    file_path = "data/feature/csv0419_1/D0NOPR_csv0419_1.csv"

    # --- 修改点 2: 输入特征范围 ---
    selected_features = "B,C,D "

    # 执行绘图
    plot_features_separate_windows(file_path, selected_features)