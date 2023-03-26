import pandas as pd

# * Alzheimer Disease and Healthy Aging Data In US.csv
# file_path = 'datasets/Alzheimer Disease and Healthy Aging Data In US.csv'
# res_path = 'result/Alzheimer5number.csv'

# * movies_dataset.csv
file_path = "datasets/movies_dataset.csv"
res_path = 'result/movies5number.csv'

def numerical_attribute(file_path,res_path):
    # todo:数值属性，给出5数概括及缺失值的个数
    df = pd.read_csv(file_path)
    # 统计数值属性
    df.describe().to_csv(res_path)


if __name__ == '__main__':
    numerical_attribute(file_path,res_path)
    
