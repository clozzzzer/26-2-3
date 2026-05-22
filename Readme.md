# data


---

# fig
特征可视化图片

---

# Result
聚类、分类结果

---

# 脚本代码
- **feature.py**  
提取所有特征
- **feature_Sequence.py**  
提取序列特征
- **KMeans_Sequence.py**  
使用序列特征对数据进行聚类
- **KMeans_Sequence_Importance.py**  
使用加权后的序列特征对数据进行分类
- **LSTM.py**  
使用深度学习方法对数据进行分类
- **pre_foldfile.py**  
对.Pcap文件进行预处理
- **RF_All.py**  
使用所有序列特征对样本进行分类
- **RF_Selected**  
在所有样本里面抽取一部分进行分类  
TARGET_CLASS_COUNT = 10  # ⬅️ 目标抽取的类别数量  
TOP_N_FEATURES = 20 # ⬅️ 特征数量
- **RF_sequence.py**  
RF_sequence.py使用随机森林算法，利用序列特征对样本进行分类  
序列特征包括excess,Range_1,Range_2,lack等
- **RF_Top_30**  
基于RF_All.py，提取了最重要的30个特征进行分类
- **visual_sequence_feature.py**  
对序列特征进行可视化

# 26-5-15
程序运行流程：

- **feature.py**  
输入数据提取特征
BASE_PATH = "./data/csv0419_1_4"  
OUTPUT_DIR = "./data/feature/csv0419_1_4"  
OUTPUT_FILE_ALL = os.path.join(OUTPUT_DIR, "All_feature_csv0419_1_4.csv")    
生成：  
   ⚠️ D1NOPR: 无数据，跳过保存。  
   ⚠️ D1PR: 无数据，跳过保存。  
   ⚠️ D0NOPR: 无数据，跳过保存。  
   ✅ D0PR: 297 条 -> ./data/feature/csv0419_1_4\D0PR_csv0419_1.csv  
   ✅ 已保存清洗后的总文件: ./data/feature/csv0419_1_4\All_Cleaned_feature_csv0419_1.csv (297 条)  
- 

