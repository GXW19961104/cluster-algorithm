import numpy as np
import random
"""
进行数据预处理
从600个数据中取出60个进行聚类分析
每一百个中提取10个
"""
data_path='synthetic_control.data'
def LoadTxt():
    """
    数据导入
    :return:矩阵
    """
    data=np.loadtxt(data_path)
    return data
def RandomPut(data):
    index=[]
    for i in range(6):                       #生成60个实验数据
        RD = [random.randint(i*100, (i+1)*100-1) for _ in range(10)]  # 生成十个随机数
        index.extend(RD)                    #在列表元素中一次性追加多值
    print(index)
    data_test=[]
    for i in range(len(index)):
        data_test.append(data[index[i]])
    return index,data_test
