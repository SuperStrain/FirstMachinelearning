# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#正规方程的优化方法对波士顿房价进行预测
def linear1():
    # 1、获取数据集
    boston=load_boston()
    # 2、划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    # 3、特征工程:标准化

    # 4、预估器流程
    # 5、模型评估

if __name__ == '__main__':
    linear1()