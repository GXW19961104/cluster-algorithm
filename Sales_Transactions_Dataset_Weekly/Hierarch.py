# -*- coding:utf-8 -*-
import numpy as np
import math as m
import pylab as pl
import time
from sklearn import metrics             #进行性能评估
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
labels_super=[-1]*811           #聚类标签
labels_true=[-1]*811
data_path='Sales_Transactions_Dataset_Weekly.csv'
def LoadTxt():
    """
    数据导入
    :return:矩阵
    """
    data=np.loadtxt(data_path,delimiter=",",skiprows=1,usecols=range(55,107))
    data=np.array(data)
    return data
def DataS():
    """
    导入数据预处理之后的数据
    :return: 矩阵
    """
    return LoadTxt()
# def label_init():
#     """
#     进行标签的初始化
#     :return:
#     """
#     del labels_true[:]
#     for i in range(6):
#         label=[i]*100
#         labels_true.extend(label)
#     return labels_true
#计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return m.sqrt(np.power(a - b, 2).sum())
#dist_min
def dist_min(Ci, Cj):
    return min(dist(i, j) for i in Ci for j in Cj)
#dist_max
def dist_max(Ci, Cj):
    return max(dist(i, j) for i in Ci for j in Cj)
#dist_avg
def dist_avg(Ci, Cj,dataset):       #Ci和Cj为两个索引
    sumCi = [0] * len(dataset[0])
    sumCj = [0] * len(dataset[0])
    for i in Ci:
        sumCi=+dataset[i]
    avgCi=sumCi/len(Ci)
    for j in Cj:
        sumCj=+dataset[j]
    avgCj=sumCj/len(Cj)
    return dist(avgCi,avgCj)

#找到距离最小的下标
def find_Min(M):
    min = 1000
    x = 0; y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j];x = i; y = j
    return (x, y, min)

#处理出距离矩阵
def dis_matrix(data):
    """
    输入数据
    :param data:
    :return:距离矩阵
    """
    npoints=len(data)
    dis_matrix=np.zeros((npoints,npoints))
    for i in range(npoints):
        for j in range(npoints):
            dis_matrix[i][j]=dis_matrix[j][i]=dist(data[i],data[j])
    return dis_matrix

#算法模型：
def AGNES(dataset, dist, k):
    #初始化C和M
    C = [];M = []
    for i in range(len(dataset)):
        Ci = []
        Ci.append(i)
        C.append(Ci)
    dis_M=dis_matrix(dataset)
    q = len(dataset)
    #合并更新
    while q > k:
        x, y, min = find_Min(dis_M)
        print(x,y,min)
        C[x].extend(C[y])
        C.remove(C[y])
        print(len(C))
        dis_M=np.zeros((len(C),len(C)))
        for i in range(len(C)):
            for j in range(len(C)):
                dis_M[i][j]=dis_M[j][i]=dist(C[i],C[j],dataset)
        q -= 1
    print(C)
    for i in range(k):
        for j in C[i]:
            labels_super[j]=i        #对标签进行赋值，确定聚类

    return C
#画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()
if __name__ == '__main__':
    start = time.clock()
    data=DataS()
    data=np.array(data)
    # labels_true=label_init()
    # labels_super=label_init()
    pca = PCA(n_components=2)  # 进行PCA降维
    newdata = pca.fit_transform(data)
    ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
    labels_super=ac.fit_predict(data)
    plt.scatter(newdata[:, 0], newdata[:, 1], c=labels_super)
    plt.show()
    print('轮廓系数' + str(metrics.silhouette_score(newdata, labels_super, metric='euclidean')))
    print('Calinski系数' + str(metrics.calinski_harabaz_score(newdata, labels_super)))
    #draw(C)
    elapsed = (time.clock() - start)
    print("用时:", elapsed)
