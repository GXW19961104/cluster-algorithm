# -*- coding:utf-8 -*-
import math as m
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics             #进行性能评估
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
createVar=locals()
labels_super=[-1]*600                   #600个数据的标签，初始值为-1，表示未进行聚类分析
labels_true=[-1]*600                    #600个数据的真是标签
global num                              #用来统计迭代次数
num=0
data_path='synthetic_control.data'
def LoadTxt():
    """
    数据导入
    :return:矩阵
    """
    data=np.loadtxt(data_path)
    return data
def DataS():
    """
    导入数据预处理之后的数据
    :return: 矩阵
    """
    return LoadTxt()
def label_init():
    """
    进行标签的初始化
    :return:
    """
    del labels_super[:]
    for i in range(6):
        label=[i]*100
        labels_super.extend(label)
    del labels_true[:]
    for i in range(6):
        label=[i]*100
        labels_true.extend(label)
    return labels_super

def dis(a,b):
    """
    进行欧式距离的计算
    :param a:
    :param b:
    :return:
    """
    return m.sqrt(np.power(a-b,2).sum())
def DTW_dis(a,b):
    """
    进行DTW距离的计算，探究DTW距离对于时间序列的影响
    :param a:
    :param b:
    :return:
    """
def SumM(index,data):
    """
    用于计算聚类中心
    :param index:
    :return:
    """
    sum=[0]*len(data[0])
    for i in range(len(index)):
        sum=sum+data[index[i]]
    return sum/len(index)

def NewNode(Cluster,data):                       #用于产生新的聚类中心
    global num                                   #全局变量的声明
    num=num+1
    New_cluster=[]
    for i in range(len(data)):                   # 聚类
        dis_mix = []
        for j in range(len(Cluster)):
            dis_mix.append(dis(data[i], Cluster[j]))
        labels_super[i] = np.argsort(dis_mix)[0]  # 确定簇标记，并进行划分,将dis_mix中的元素从小到大排列，提取其对应的索引
    print('经过' + str(num) + '次迭代，聚类效果ARI='+str(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_super)))
    # plt.scatter(data[:, 0], data[:, 1], c=labels_super)
    # plt.show()
    for i in range(len(Cluster)):               # 聚类中心点迭代
        labels_super1 = np.array(labels_super)  # 将列表转化为矩阵
        index = np.argwhere(labels_super1 == i)
        New_cluster.append(SumM(index,data))    # 生成6个新聚类中心
    return New_cluster

def Kmeans(K,data):
    """
    K均值算法的具体实现，根据西瓜书上的伪代码
    :param K: 聚类个数
    :param data: 数据
    :return: 每个数据的簇标记
    """
    # plt.scatter(data[:, 0], data[:, 1], c=labels_super)
    # plt.show()
    First_C =[]       #初始聚类点
    C=[random.randint (0,len(data)-1)for _ in range(K)]    #生成初始聚类中心点
    for i in range(len(C)):
        First_C.append(data[C[i]])
    New_cluster1=NewNode(First_C,data)
    err=np.array(New_cluster1) -np.array(First_C)
    Jud=1           #用于判断循环是否继续进行
    while Jud:
        New_cluster2=NewNode(New_cluster1,data)
        err=np.array(New_cluster2) -np.array(New_cluster1)
        if (np.array(New_cluster2) ==np.array(New_cluster1)).all():
            Jud=0
        else:
            Jud=1
        New_cluster1=New_cluster2
    print('经过'+str(num)+'次迭代，聚类完成')
    print(labels_super)
    return labels_super
def KM(data):
    """
    K-means算法
    :param index:
    :param data:
    :return:

    """
    pca = PCA(n_components=2)  # 进行PCA降维
    newdata = pca.fit_transform(data)
    labels_super=Kmeans(6, newdata)
    plt.scatter(newdata[:, 0], newdata[:, 1], c=labels_super)
    print(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_super))  #进行ARI性能评估，取值[-1,1]越接近1，性能越好
    plt.show()
#主函数
if __name__ == '__main__':
    labels_super=label_init()
    data=DataS()
    KM(data)



