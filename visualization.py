import pandas as pd
import matplotlib.pyplot as plt

# * Alzheimer Disease and Healthy Aging Data In US.csv
# file_path = 'datasets/Alzheimer Disease and Healthy Aging Data In US.csv'
# hist_res_path = 'result/AlzheimerDataHistogram.jpg'
# boxp_res_path = 'result/AlzheimerDataBoxplot.jpg'

# * movies_dataset.csv
file_path = 'datasets/movies_dataset.csv'
hist_res_path = 'result/moviesDataHistogram.jpg'
boxp_res_path = 'result/moviesDataBoxplot.jpg'

def histogram(file_path, res_path):
    """
    todo: 画直方图
    """
    df = pd.read_csv(file_path)
    df.hist(bins=13, figsize=(10, 10))
    # plt.savefig(res_path)
    # this line will show the figure
    plt.show()


def boxplot(file_path, res_path):
    """
    todo: 画盒图（箱型图）
    :param file_path:
    :param res_path:
    :return:
    """
    df = pd.read_csv(file_path)
    index = df.dtypes
    numerical_attribute = []
    for i, v in index.iteritems():
        if v == 'int64' or v == 'float64':
            numerical_attribute.append(i)
    data = []

    for i in range(len(numerical_attribute)):
        # df.boxplot(column=item)
        data.append(df[numerical_attribute[i]])
        plt.subplot(2, 3, i + 1)
        # plt.title(numerical_attribute[i])
        df[numerical_attribute[i]].to_frame().boxplot()
        # plt.boxplot(df[numerical_attribute[i]])
    plt.tight_layout(pad=1.08)
    # plt.savefig(res_path)
    plt.show()


if __name__ == '__main__':
    histogram(file_path, hist_res_path)
    boxplot(file_path, boxp_res_path)
