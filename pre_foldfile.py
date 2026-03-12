import pyshark
from time import time as timestamp
import csv
import os

# 主逻辑：遍历文件夹并处理每个 PCAP 文件
dataset_folder_path = './data/dataset0423'
csv_base_path = r'./data/dataset0423_csv'

# 定义帧类型和子类型的映射
FRAME_TYPES = {
    (0, 4): "Probe Request",
    (0, 5): "Probe Response",
    (0, 11): "Authentication",
    (0, 0): "Association Request",
    (0, 1): "Association Response",
}

FRAME_TYPES_To_Int = {
    'Probe Request': 1,
    'Probe Response': 2,
    "Authentication": 3,
    "Association Request": 4,
    "Association Response": 5,
    "EAPOL": 6
}


def process_pcap_file(pcap_file):
    """
    处理单个 PCAP 文件，提取帧信息并返回时间、目标地址和类型。
    """
    time, type_, dst = [], [], []
    first_time = None

    try:
        capture = pyshark.FileCapture(pcap_file)

        for i, packet in enumerate(capture):
            try:
                # 检查是否是无线报文
                if "WLAN" in packet:
                    wlan_layer = packet.wlan

                    # 提取 fc_type 和 fc_subtype 字段值
                    if "fc_type" in wlan_layer.field_names and "fc_subtype" in wlan_layer.field_names:
                        frame_type_raw = wlan_layer.get_field_value("fc_type")
                        frame_subtype_raw = wlan_layer.get_field_value("fc_subtype")

                        # 清理字段值并转换为整数
                        try:
                            frame_type = int(str(frame_type_raw).strip(), 0)
                            frame_subtype = int(str(frame_subtype_raw).strip(), 0)

                            # 判断帧类型
                            frame_key = (frame_type, frame_subtype)
                            frame_name = FRAME_TYPES.get(frame_key, None)

                            # 如果是 Data Frame，进一步检查是否是 EAPOL 帧
                            if 'EAPOL' in packet:
                                frame_name = 'EAPOL'

                            if frame_name:
                                # 提取源 MAC 和目标 MAC 地址
                                dest_mac = getattr(wlan_layer, "da", "N/A")

                                # 记录类型和目标地址
                                if frame_name in FRAME_TYPES_To_Int:
                                    type_.append(FRAME_TYPES_To_Int[frame_name])
                                    dst.append(1 if dest_mac == 'ff:ff:ff:ff:ff:ff' else 0)

                                # 记录时间戳
                                unix_timestamp = float(packet.sniff_time.timestamp())
                                if first_time is None:
                                    first_time = unix_timestamp
                                    time.append(0.0)
                                else:
                                    time.append(unix_timestamp - first_time)

                        except ValueError:
                            print(f"Packet {i + 1}: Failed to convert fc_type or fc_subtype to integer.")
            except AttributeError as e:
                print(f"Error processing packet {i + 1}: {e}")

        capture.close()

    except Exception as e:
        print(f"Error processing file {pcap_file}: {e}")

    return time, dst, type_

def write_to_csv(csv_file, time, dst, type_):
    """
    将时间、目标地址和类型写入 CSV 文件。
    """
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['Time', 'Dst', 'Type'])

        # 写入数据行
        for tm, d, t in zip(time, dst, type_):
            writer.writerow([tm, d, t])

for root, dirs, files in os.walk(dataset_folder_path):
    print(f"当前目录: {root}")
    print(f"子目录列表: {dirs}")
    print(f"文件列表: {files}")

    # 构造对应的 CSV 文件夹路径
    parts = root.split("\\")
    csv_folder_path = os.path.join(csv_base_path, *parts[-3:])
    os.makedirs(csv_folder_path, exist_ok=True)  # 确保文件夹存在

    for file_name in files:
        if file_name.endswith('.pcap'):  # 只处理 .pcap 文件
            pcap_file = os.path.join(root, file_name)
            print(f"Processing file: {pcap_file}")

            # 处理 PCAP 文件
            time, dst, type_ = process_pcap_file(pcap_file)

            # 构造 CSV 文件路径
            csv_file = os.path.join(csv_folder_path, file_name[:-5] + '.csv')

            # 写入 CSV 文件
            write_to_csv(csv_file, time, dst, type_)

    print(f"{root} DONE")
    print('-' * 50)

print('All done!')