# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# 优点：
# 简单，易于理解，易于实现，无需训练
# 缺点：
# 懒惰算法，对测试样本分类时的计算量大，内存开销大
# 必须指定K值，K值选择不当则分类精度不能保证
# 使用场景：小数据场景，几千～几万样本，具体场景具体业务去测试
# 案例1：鸢尾花种类预测
# （1）获取数据
# （2）数据集划分
# （3）特征工程 标准化处理
# （4）KNN预估器流程
# （5）模型评估

def knn_iris():
    """
    使用knn算法对鸢尾花进行分类
    :return:
    """
    # （1）获取数据
    iris=load_iris()

    # （2）数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=22)
    # （3）特征工程 标准化处理
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    # （4）KNN预估器流程
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    # （5）模型评估
    #方法1：直接比对真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值\n",y_test==y_predict)
    #方法2：计算准确率
    score=estimator.score(x_test,y_test)
    print('准确率为\n',score)

if __name__ == '__main__':
    knn_iris()