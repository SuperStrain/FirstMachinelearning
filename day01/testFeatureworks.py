# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm
import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

#特征工程是：使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程

#sklearn 特征工程
#pandas 数据清洗、数据处理

#特征工程内容：特征抽取、特征预处理、特征降维

#1、特征抽取：
# 机器学习算法——统计方法——数学公式（文本类型->数值）
#(1)字典特征提取
def dict_demo():
    """
    字典特征抽取
    应用场景：
    1）pclass,sex 数据集当中类别特征较多
        1、将数据集特征->字典类型
        2、DictVectorizer转换
    2）本身拿到的就是字典数据类型
    """
    data=[{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    #(1)、实例化一个转换器类
    transfer=DictVectorizer(sparse=False)#默认是返回稀疏(sparse)矩阵：元组表示坐标，后面跟内容，只表示非零元素
                                        #稀疏(sparse)矩阵节省内存空间
    #(2)、调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data为：\n',data,'转换为：\n',data_new)
    print('特征名字：\n',transfer.get_feature_names())
    return None

#(2)文本特征提取
def text_demo():
    """
    文本特征提取CountVectorizer
    :return:
    """
    data=["life is short,i like like python", "life is too long,i dislike python"]
    #(1)、实例化一个转换器类
    transfer=CountVectorizer(stop_words=['is','too'])
    #(2)、调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data为：\n',data,'转换为：\n',data_new.toarray())#转换为正常矩阵
    print('对应的意义：\n',transfer.get_feature_names())

def cut_words(text):
    """
    将传入的中文进行分词 “我爱北京天安门”-->“我 爱 北京 天安门”
    :param text:
    :return:
    """
    text=" ".join(list(jieba.cut(text)))#将分词结果转换为list再转换为str
    return text

#文本特征提取（中文）
def Chinese_text_demo():
    """
    中文文本特征提取TfidfVectorizer
    :return:
    """
    data=['今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
          '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
          '如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']
    #(1)、实例化一个转换器类
    transfer=CountVectorizer()
    #(2)、调用fit_transform()
    for i in range(len(data)):
        data[i]=cut_words(data[i])
    data_new=transfer.fit_transform(data)
    print('data为：\n',data,'转换为：\n',data_new.toarray())#转换为正常矩阵
    print('对应的意义：\n',transfer.get_feature_names())

#(3)Tf-idf文本特征提取：
#用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
def tfidf_demo():
    """
    中文文本特征提取CountVectorizer
    :return:
    """
    data=['今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
          '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
          '如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']
    #(1)、实例化一个转换器类
    transfer=TfidfVectorizer(stop_words=['一种','只用'])
    #(2)、调用fit_transform()
    for i in range(len(data)):
        data[i]=cut_words(data[i])
    data_new=transfer.fit_transform(data)
    print('data为：\n',data,'转换为：\n',data_new.toarray())#转换为正常矩阵
    print('对应的意义：\n',transfer.get_feature_names())


# 2、特征预处理：通过一些转换函数将 特征数据 转换成更加适合算法模型的特征数据过程
# 原因：特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，
# 容易影响（支配）目标结果，使得一些算法无法学习到其它的特征。

#（1）归一化
# 特点：最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。
def minmax_demo():
    """
    归一化
    :return:
    """
    # 1、获取数据
    data=pd.read_csv('accessory/dating.txt')
    data=data.iloc[:,:3]#每行都要，列只选前三列
    #2、实例化一个转换器类
    transfer=MinMaxScaler(feature_range=[2,3])#输出值的范围
    #3、调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data为：\n', data, '\n','转换为：\n', data_new)

    return None

#（2）标准化(一般较为常见)
# 特点：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小。
def standard_demo():
    # 1、获取数据
    data=pd.read_csv('accessory/dating.txt')
    data=data.iloc[:,:3]#每行都要，列只选前三列
    #2、实例化一个转换器类
    transfer=StandardScaler()#输出值的范围
    #3、调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data为：\n', data, '\n','转换为：\n', data_new)

    return None


#3、特征降维
# 降维是指在某些限定条件下，降低随机变量(特征)个数，得到一组“不相关”主变量的过程（去掉冗余），
# 旨在从原有特征中找出主要特征

#（1）Filter(过滤式)：主要探究特征本身特点、特征与特征和目标值之间关联
# 方差选择法：低方差特征过滤
# 相关系数
def variance_demo():
    """低方差特征过滤"""
    #1、获取数据
    data=pd.read_csv('accessory/factor_returns.csv')
    data=data.iloc[:,1:-2]
    # 2、实例化一个transfer转换器
    transfer=VarianceThreshold(threshold=5)#方差阈值改为5
    # 3、调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data为：\n', data, '\n','转换为：\n', data_new,data_new.shape)

    # 计算某两个变量的相关性
    r1=pearsonr(data['pe_ratio'],data['pb_ratio'])
    print("相关系数为：\n",r1)
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print('revenue与total_expense之间的相关性为：\n',r2)


#主成分分析(PCA) 降维
# 定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
# 作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
# 应用：回归分析或者聚类分析当中
def pca_demo():
    #1、获取数据
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]

    # 2、实例化一个transfer转换器
    transfer=PCA(n_components=2)#小数：表示保留百分之多少的信息 整数：减少到多少特征
    # 3、调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data为：\n', data, '\n','转换为：\n', data_new,data_new.shape)




if __name__=='__main__':
    # dict_demo()
    # text_demo()
    # print(cut_words("我爱北京天安门"))#测试分词函数
    # Chinese_text_demo()
    # tfidf_demo()
    # minmax_demo()
    # standard_demo()
    # variance_demo()
    pca_demo()

