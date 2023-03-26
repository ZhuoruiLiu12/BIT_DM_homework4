import pandas as pd
import os
import numpy as np

# file_path_list = ['datasets/Alzheimer Disease and Healthy Aging Data In US.csv','datasets/movies_dataset.csv']
# res_path_list = ['result/AlzheimerDataFrequency.csv','result/moviesDataFrequency.csv']

# *  Alzheimer Disease and Healthy Aging Data In US.csv
# file_path = 'datasets/Alzheimer Disease and Healthy Aging Data In US.csv'
# res_path = 'result/AlzheimerDataFrequency.csv'

# * movies_dataset.csv
file_path = "datasets/movies_dataset.csv"
res_path = 'result/moviesDataFrequency.csv'


def nominal_attribute_frequency(file_path,res_path):
    # todo: 标称属性，给出每个可能取值的频数
    df = pd.read_csv(file_path,sep=',')
    # * nominal attribute
    # * nominal dict 标称属性字典
    nominal_attribute = df.columns
    nominal_data_frequency = {}
    for column in nominal_attribute:
        df_frequency = df[column].value_counts()
        nominal_data_frequency[column] = df_frequency

    # * dataframe 2 csv
    # PATH = 'result/AlzheimerDataFrequency.csv'
    for key in nominal_data_frequency.keys():
        if os.path.exists(res_path):
            nominal_data_frequency[key].to_csv(res_path,mode='a+',sep=',')
        else:
            nominal_data_frequency[key].to_csv(res_path,sep=',')

    # *修改csv文件内容
    df2 = pd.read_csv(res_path,header=None)
    index = df2[0].isnull()
    for i, v in index.iteritems():
        if v == True:
            temp = df2[1][i]
            df2[0][i] = temp
            df2[1][i] = np.nan
    df2.to_csv(res_path,index=False,header=False)
    df2.info()


if __name__ == '__main__':
    nominal_attribute_frequency(file_path,res_path)

# print(nominal_data_frequency)

