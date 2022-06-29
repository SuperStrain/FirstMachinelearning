# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# #获取小规模的数据集，数据包含在datasets里
# sklearn.datasets.load_*()
#
# #获取大规模的数据集，需要在网络上下载，函数的第一个参数是data_home，表示数据集下载的目录
# sklearn.datasets.fetch_*()



#展示鸢尾花数据集
#数据集用途：
#1、训练模型
#2、测试模型：评估模型的效果 占20%~30%左右
def datasets_demo():
    iris=load_iris() #鸢尾花数据集
    # print("鸢尾花数据集：\n",iris)
    # print("查看数据集描述：\n",iris['DESCR'])
    # print("查看特征值的名字：\n",iris.feature_names)
    # print("查看特征值：\n",iris.data,iris.data.shape)

    #数据集划分:
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    #x数据集的特征值，y数据集的特征值，test_size测试集的大小，一般为float
    #random_state 随机数种子，不同的种子随机采样的结果不同，相同的种子采样结果相同
    #return 训练集特征值，测试集特征值，训练集目标值，测试集目标值
    print("训练值的特征值\n",x_train,x_train.shape)



if __name__=='__main__':
    datasets_demo()