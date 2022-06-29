# -*- codeing= utf-8 -*-
# @Time : 2022/5/15 11:12
# @Author : Yina
# @File : f.py
# @Software: PyCharm

#这个得用服务器，电脑太难跑

import pandas as pd

def test_demo():
    # 1、获取数据
    aisles=pd.read_csv('./instacart/aisles.csv')
    orders = pd.read_csv('./instacart/orders.csv')
    order_products__prior = pd.read_csv('./instacart/order_products__prior.csv')
    products = pd.read_csv('./instacart/products.csv')
    # 2、合并表
    tab1 = pd.merge(orders, order_products__prior, on=["order_id", "order_id"])
    print(tab1)
    tab2 = pd.merge(tab1, products, on=["product_id", "product_id"])
    print(tab2)
    tab3 = pd.merge(tab2, aisles, on=["aisle_id", "aisle_id"])
    print(tab3)
    # 3、找到user_id和aisle之间的关系
    # 4、PCA降维

if __name__ == '__main__':
    test_demo()