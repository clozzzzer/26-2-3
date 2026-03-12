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
- **RF_sequence.py**  
RF_sequence.py使用随机森林算法，利用序列特征对样本进行分类  
序列特征包括excess,Range_1,Range_2,lack等
- **RF_Top_30**  
基于RF_All.py，提取了最重要的30个特征进行分类
- **visual_sequence_feature.py**  
对序列特征进行可视化