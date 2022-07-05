# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def titanic_main():
    #1、获取数据
    titanic=pd.read_csv('titanic.csv')
    # print(titanic)

    #筛选特征值和目标值
    x=titanic[['pclass','age','sex']]
    y=titanic['survived']

    #2、数据处理
    # 1）缺失值处理
    x['age'].fillna(x['age'].mean(),inplace=True)
    # 2）转换成字典
    x=x.to_dict(orient='records')

    # 3、数据集划分
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22)#x特征值，y目标值

    # 4、特征工程：字典特征提取
    transfer=DictVectorizer()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 5、决策树算法预测
    # 1）决策树预估器
    estimator=DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train,y_train)

    # 2）模型评估
    # 方法1：直接比对真实值的预测值
    y_predict=estimator.predict(x_test)
    print('y_predict:\n',y_predict)
    print('直接比对真实值的预测值:\n',y_test==y_predict)

    # 方法2：计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为：\n",score)

    #可视化决策树
    export_graphviz(estimator,out_file='titanic_tree.dot',feature_names=transfer.get_feature_names())



if __name__ == '__main__':
    titanic_main()
