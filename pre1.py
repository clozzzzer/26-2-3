import pyshark
from time import time as timestamp  # 导入函数
import csv
import os
'''
pre1.py是对一个文件进行预处理的代码
pre2.py是对一个文件夹中所有的文件进行遍历处理的代码
'''

# 定义帧类型和子类型的映射
FRAME_TYPES = {
    (0, 4): "Probe Request",
    (0, 5): "Probe Response",
    (0, 11): "Authentication",
    (0, 0): "Association Request",
    (0, 1): "Association Response",
    (2, 8): "Data Frame"  # 初始定义为数据帧
}

FRAME_TYPES_To_Int = {
    'Probe Request' : 1,
    'Probe Response' : 2,
    "Authentication" : 3,
    "Association Request" : 4,
    "Association Response" : 5,
    "EAPOL" : 6
}

# 读取 PCAP 文件
# 指定要遍历的文件夹路径
pcap_folder_path = 'D:\desk\BiShe\dataset\dataset0419\dataset'

# 遍历文件夹
for root, dirs, files in os.walk(pcap_folder_path):
    print(f"当前目录: {root}")
    print(f"子目录列表: {dirs}")
    print(f"文件列表: {files}")
    for file_name in files:
        pcap_file = os.path.join(root, file_name)  # 获取文件完整路径
        print(f"文件路径: {pcap_file}")
        capture = pyshark.FileCapture(pcap_file)
        time = []
        type = []
        dst = []
        first_time = int(0)
        # 遍历每个数据包并打印基本信息
        for i, packet in enumerate(capture):
            try:
                # 检查是否是无线报文
                if "WLAN" in packet:
                    wlan_layer = packet.wlan

                    # 打印 fc_type 和 fc_subtype 字段值
                    if "fc_type" in wlan_layer.field_names and "fc_subtype" in wlan_layer.field_names:
                        frame_type_raw = wlan_layer.get_field_value("fc_type")
                        frame_subtype_raw = wlan_layer.get_field_value("fc_subtype")

                        # 清理字段值并转换为整数
                        try:
                            frame_type = int(str(frame_type_raw).strip(), 0)
                            frame_subtype = int(str(frame_subtype_raw).strip(), 0)

                            # 判断帧类型
                            frame_key = (frame_type, frame_subtype)
                            if frame_key in FRAME_TYPES:
                                #print(f"Packet {i + 1}:")
                                if frame_key != (2,8):
                                    frame_name = FRAME_TYPES[frame_key]

                                # 如果是 Data Frame，进一步检查是否是 EAPOL 帧
                                if frame_key == (2, 8):
                                    frame_control_field = None
                                    if not frame_control_field and hasattr(wlan_layer, "fc"):
                                        frame_control_field = wlan_layer.get_field_value("fc")
                                    if frame_control_field=='0x00008801' or frame_control_field == '0x00008802':
                                        frame_name = 'EAPOL'


                                #print(f"{frame_name} Detected!")

                                # 提取源 MAC 和目标 MAC 地址
                                source_mac = getattr(wlan_layer, "sa", "N/A")
                                dest_mac = getattr(wlan_layer, "da", "N/A")
                                #print(f"Source MAC: {source_mac}, Destination MAC: {dest_mac}")

                                if frame_name in FRAME_TYPES_To_Int:
                                    type.append(FRAME_TYPES_To_Int[frame_name])
                                    if dest_mac == 'ff:ff:ff:ff:ff:ff':
                                        dst.append(1)
                                    else:
                                        dst.append(0)

                                # 打印时间戳
                                #timestamp = packet.sniff_time
                                unix_timestamp = float(packet.sniff_time.timestamp())
                                if not time:
                                    first_time = unix_timestamp
                                    time.append(float(0))
                                else:
                                    t = unix_timestamp - first_time
                                    time.append(t)
                                #print(unix_timestamp)
                                # if not time:
                                #     first_time = timestamp
                                #     time.append(int(0))
                                #print("-" * 50)
                        except ValueError:
                            print(f"Packet {i + 1}:")
                            print("Failed to convert fc_type or fc_subtype to integer.")
                            print("-" * 50)

            except AttributeError as e:
                print(f"Error processing packet: {e}")

            #print(time)
        #print(type)
        #print(dst)
        #print(time)
        print('time:',time)
        # 写入 CSV 文件
        parts = pcap_folder_path.split("\\")
        csv_path = "D:\desk\毕设\dataset\csv0419"
        csv_path = csv_path + '\\' + parts[-3] +  '\\' + parts[-2] + '\\' + parts[-1]
        csv_file = os.path.join(csv_path, file_name)
        csv_file = csv_file[:-5] + '.csv'
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            writer.writerow(['Time', 'Dst', 'Type'])

            # 写入数据行
            for tm, d, t in zip(time,dst,type):
                writer.writerow([tm, d, t])
        #print(f"数据已成功写入 {csv_file}")

print('done')