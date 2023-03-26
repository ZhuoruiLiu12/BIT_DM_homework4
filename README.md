# BIT_DM_homework4
DM homework for week 4

##homework requirement：
1. 数据摘要和可视化
- 数据摘要
 >标称属性，给出每个可能取值的频数
 >数值属性，给出5数概括及缺失值的个数

- 数据可视化
>使用直方图、盒图等检查数据分布及离群点

2. 数据缺失的处理
观察数据集中缺失数据，分析其缺失的原因。分别使用下列四种策略对缺失值进行处理:

- 将缺失部分剔除
- 用最高频率值来填补缺失值
- 通过属性的相关关系来填补缺失值
- 通过数据对象之间的相似性来填补缺失值
注意：在处理后完成，要对比新旧数据集的差异。
-----
##datasets:
- [Alzheimer Disease and Healthy Aging Data in US](https://www.kaggle.com/datasets/ananthu19/alzheimer-disease-and-healthy-aging-data-in-us)
- [Movies Dataset from Pirated Sites](https://www.kaggle.com/datasets/arsalanrehman/movies-dataset-from-piracy-website)
-----
## how to run the code
1. 将datasets从上述链接中下载下来，放在datasets目录下
2. 每个代码文件中注意path变量的设置(包括数据文件路径和结果保存路径)，设置完之后运行代码即可得到想要的分析结果
3. 代码文件说明
> - nominal_attribute.py:标称属性，给出每个可能取值的频数
> - numerical_attribute.py:数值属性，给出5数概括及缺失值的个数
> - visualization.py:使用直方图、盒图等检查数据分布及离群点
> - handing_missing_data.py：处理数据缺失值并对比新旧数据集差异