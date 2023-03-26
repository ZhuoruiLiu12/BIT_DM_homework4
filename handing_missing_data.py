import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer


path = 'datasets/movies_dataset.csv'


def simi_fill(file_path, k=30):
    """
    todo:通过数据对象之间的相似性来填补缺失值 使用knn聚类求相似对象
    :param file_path:
    :return:
    """
    # KNN
    df = pd.read_csv(file_path)
    df_cp = df.copy(deep=True)
    numerical_attr = numerical_index(df)
    # df_cp[numerical_attr].values
    # imputed_training = fast_knn(df_cp[numerical_attr].values, k=k)
    # imputed_training = pd.DataFrame(data=imputed_training, columns=numerical_attr)
    # df_cp[numerical_attr] = imputed_training[numerical_attr]
    # knn = KNeighborsClassifier(n_neighbors=k)
    for i in numerical_attr:
        if df[i].isnull().all():
            numerical_attr.remove(i)
    imputer = KNNImputer(n_neighbors=k)
    df_filled = imputer.fit_transform(df[numerical_attr])
    df_filled = pd.DataFrame(df_filled, columns=numerical_attr)
    df_cp[numerical_attr] = df_filled[numerical_attr]
    return df, df_cp


def miss_index(df: pd.DataFrame) -> list:
    """
    todo: 返回数据集中有缺失的属性 注意属性列全空的情况
    :param df:
    :return:
    """
    res = []
    for i, v in df.isnull().any().iteritems():
        if v:
            res.append(i)
    return res


def numerical_index(df: pd.DataFrame) -> list:
    """
    todo: 求出数据集中的数值属性
    :param df:
    :return:
    """
    numerical_attribute = []
    for i, v in df.dtypes.iteritems():
        if v == 'int64' or v == 'float64':
            numerical_attribute.append(i)
    return numerical_attribute


def attr_corr_fill(file_path):
    df = pd.read_csv(file_path)
    miss_attr = miss_index(df)
    attr = df.columns.tolist()
    comp_attr = []
    for item in attr:
        if item not in miss_attr:
            comp_attr.append(item)

    def set_miss_value(df: pd.DataFrame, comp_attr: list):
        """
        todo:使用RF算法 根据属性相关性补全缺失值
        :param df:
        :param comp_attr:
        :return:
        """
        enc_label = OrdinalEncoder()
        # enc_label = LabelEncoder()
        enc_fea = OrdinalEncoder()
        missing_index = comp_attr[0]

        train_df = df[comp_attr]
        known_values = np.array(train_df[train_df[missing_index].notnull()])
        unknown_values = np.array(train_df[train_df[missing_index].isnull()])

        y = known_values[:, 0].reshape(-1, 1).astype(str)
        # print(y.dtype)
        # str_list = ['Low_Confidence_Limit', 'High_Confidence_Limit']
        # if missing_index in str_list:
        #     y.astype(str)
        enc_label.fit(y)
        # enc_label.fit_transform(y)
        y = enc_label.transform(y)

        x = known_values[:, 1:].astype(str)
        x_test = unknown_values[:, 1:].astype(str)
        x_all = np.row_stack((x, x_test))
        enc_fea.fit(x_all)
        x = enc_fea.transform(x)

        # fit
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(x, y.ravel())
        # predict
        predict_value = rfr.predict(enc_fea.transform(unknown_values[:, 1:].astype(str)))
        predict_value = enc_label.inverse_transform(predict_value.reshape(-1, 1))

        df.loc[(df[missing_index].isnull()), missing_index] = predict_value
        return df

    df_cp = df.copy(deep=True)
    for i in range(0, len(miss_attr)):
        if not df[miss_attr[i]].isnull().all():
            comp_attr.insert(0, miss_attr[i])
            df_cp = set_miss_value(df_cp, comp_attr)
    # if not df[miss_attr[4]].isnull().all():
    #     pass,ml,
    #     print(0)
    # comp_attr.insert(0, miss_attr[0])
    # df_cp = set_miss_value(df_cp, comp_attr)

    return df, df_cp


def delete_missing_data(file_path):
    """
    todo:删除缺失数据
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path)
    df_new = df.dropna()
    # print(df_new.empty)
    return df, df_new


def max_frequency(file_path):
    """
    todo:用最高频率填补缺失值
    :return:
    """
    df = pd.read_csv(file_path)
    # columns = df.columns.values.tolist()
    for i, v in df.isnull().any().iteritems():
        if v and (not df[i].value_counts().empty):
            max_value = df[i].value_counts().idxmax()
            df[i].fillna(max_value, inplace=True)
    df_old = pd.read_csv(file_path)
    return df_old, df


def data_comparison(origin_data: pd.DataFrame, new_data: pd.DataFrame):
    """
    todo:处理前后数据对比
    :param origin_data:
    :param new_data:
    :return:
    """
    if new_data.empty:
        print("new data is empty!")
        # df_difference = origin_data.compare(new_data)
    else:
        df_difference = origin_data.compare(new_data)
    origin_data.info()
    new_data.info()
    numerical_attribute = numerical_index(origin_data)
    for item in numerical_attribute:
        df_plot = pd.DataFrame({'origin_data': origin_data[item], 'new_data': new_data[item]})
        # plt.hist(origin_data[item].values, alpha=0.5, label="origin data")
        # plt.hist(new_data[item].values, alpha=0.5, label='new data')
        df_plot.hist(bins=12)
        plt.xlabel(item)
        # plt.title(item)
        plt.show()
    # df_plot = pd.DataFrame({'origin_data':origin_data, 'new_data':new_data},index=[0])
    # df_plot.hist(alpha=0.5)


if __name__ == '__main__':
    # 删除缺失数据
    origin_data1, new_data1 = delete_missing_data(path)
    data_comparison(origin_data1, new_data1)
    # 用最高频率值来填补缺失值
    origin_data2, new_data2 = max_frequency(path)
    data_comparison(origin_data2, new_data2)
    # 通过属性的相关关系来填补缺失值
    origin_data3, new_data3 = attr_corr_fill(path)
    data_comparison(origin_data3, new_data3)
    # 通过数据对象之间的相似性来填补缺失值
    origin_data4, new_data4 = simi_fill(path)
    data_comparison(origin_data4, new_data4)
    # print(0)
