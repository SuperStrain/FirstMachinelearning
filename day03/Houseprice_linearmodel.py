# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error

#正规方程的优化方法对波士顿房价进行预测
def linear1():
    # 1、获取数据集
    boston=load_boston()
    # 2、划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)
    # 3、特征工程:标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 4、预估器流程
    estimator=LinearRegression()
    estimator.fit(x_train,y_train)
    print("正规方程-权重系数为：\n",estimator.coef_)
    print("正规方程-偏置值为：\n",estimator.intercept_)
    # 5、模型评估
    y_predict=estimator.predict(x_test)
    print("预测房价为：\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print('正规方程-均方误差为：\n',error)


#梯度下降的优化方法对波士顿房价进行预测
def linear2():
    # 1、获取数据集
    boston=load_boston()
    # 2、划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)
    # 3、特征工程:标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 4、预估器流程
    estimator=SGDRegressor()
    estimator.fit(x_train,y_train)
    print('特征数量：\n',boston.data.shape)
    print("梯度下降-权重系数为：\n",estimator.coef_)
    print("梯度下降-偏置值为：\n",estimator.intercept_)
    # 5、模型评估
    y_predict=estimator.predict(x_test)
    print("预测房价为：\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print('梯度下降-均方误差为：\n',error)

#线性回归的改进-岭回归
def linear3():
    # 1、获取数据集
    boston=load_boston()
    # 2、划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)
    # 3、特征工程:标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 4、预估器流程
    estimator=Ridge()
    estimator.fit(x_train,y_train)
    print('特征数量：\n',boston.data.shape)
    print("岭回归-权重系数为：\n",estimator.coef_)
    print("岭回归-偏置值为：\n",estimator.intercept_)
    # 5、模型评估
    y_predict=estimator.predict(x_test)
    print("预测房价为：\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print('岭回归-均方误差为：\n',error)


if __name__ == '__main__':
    linear1()
    linear2()
    linear3()