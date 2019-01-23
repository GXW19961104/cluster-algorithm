# -*- coding:utf-8 -*-
import numpy as np
import math as m
import queue
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearn import metrics             #进行性能评估
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data_path = "plants3.data.txt"
NOISE = 0
UNASSIGNED = -1
def load_data():
    points = np.loadtxt(data_path, skiprows=1)
    print(np.shape(points))  #导入数据的格式为（3382,69）
    return points


def dist(a, b):
    """
    计算两个向量的距离
    :param a: 向量1
    :param b: 向量2
    :return: 距离
    """
    return m.sqrt(np.power(a-b, 2).sum())


def neighbor_points(data, pointId, radius):
    """
    得到邻域内所有样本点的Id
    :param data: 样本点
    :param pointId: 核心点
    :param radius: 半径
    :return: 邻域内所用样本Id
    """
    points = []
    for i in range(len(data)):
        if dist(data[i], data[pointId]) < radius:
            points.append(i)
    return np.asarray(points)


def to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
    """
    判断一个点是否是核心点，若是则将它和它邻域内的所用未分配的样本点分配给一个新类
    若邻域内有其他核心点，重复上一个步骤，但只处理邻域内未分配的点，并且仍然是上一个步骤的类。
    :param data: 样本集合
    :param clusterRes: 聚类结果
    :param pointId:  样本Id
    :param clusterId: 类Id
    :param radius: 半径
    :param minPts: 最小局部密度
    :return:  返回是否能将点PointId分配给一个类
    """
    points = neighbor_points(data, pointId, radius)
    points = points.tolist()

    q = queue.Queue()
    if len(points) < minPts:
        clusterRes[pointId] = NOISE
        return False
    else:

        clusterRes[pointId] = clusterId        #对该点进行赋值
    for point in points:
        if clusterRes[point] == UNASSIGNED:
            q.put(point)
            clusterRes[point] = clusterId     #对该点的密度直达进行赋类
    print(clusterRes)
    while not q.empty():                      #寻找该点的的密度相连 对队列的使用，这个人真的秀
        neighborRes = neighbor_points(data, q.get(), radius)
        if len(neighborRes) >= minPts:                      # 核心点
            for i in range(len(neighborRes)):
                resultPoint = neighborRes[i]
                if clusterRes[resultPoint] == UNASSIGNED:
                    q.put(resultPoint)
                    clusterRes[resultPoint] = clusterId
                elif clusterRes[clusterId] == NOISE:
                    clusterRes[resultPoint] = clusterId
    return True


def dbscan(data, radius, minPts):
    """
    扫描整个数据集，为每个数据集打上核心点，边界点和噪声点标签的同时为
    样本集聚类
    :param data: 样本集
    :param radius: 半径
    :param minPts:  最小局部密度
    :return: 返回聚类结果， 类id集合
    """
    clusterId = 1
    nPoints = len(data)
    clusterRes = [UNASSIGNED] * nPoints
    print(clusterRes)
    for pointId in range(nPoints):
        if clusterRes[pointId] == UNASSIGNED:
            if to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
                clusterId = clusterId + 1
    return np.asarray(clusterRes), clusterId


def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')


if __name__ == '__main__':
    start = time.clock()
    data = load_data()
    cluster = np.asarray(data)
    clusterRes, clusterNum = dbscan(data, 4, 50)
    pca = PCA(n_components=2)  # 进行PCA降维
    newdata = pca.fit_transform(data)
    print('轮廓系数' + str(metrics.silhouette_score(newdata, clusterRes, metric='euclidean')))
    print('Calinski系数' + str(metrics.calinski_harabaz_score(newdata, clusterRes)))
    elapsed = (time.clock() - start)
    print("用时", elapsed)
    plt.scatter(newdata[:, 0], newdata[:, 1], c=clusterRes)
    plt.show()
    num = []
    for i in range(clusterNum):
        print(np.sum(clusterRes == i))
        num.append(np.sum(clusterRes == i))
    plt.bar(range(clusterNum), num)
    plt.show()
