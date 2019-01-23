# -*- coding:utf-8 -*-
#用于数据处理和数据清洗  本数据集中有69个地区,将生成70
import numpy as np
alien={}
def get_Data():
    a = 0
    with open('plants2.data.txt','w+')as file_object1:
        file_object1.truncate()
        file_object1.close()
    with open('plants.data.txt')as file_object2:
        for line in file_object2:
            Array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0,0]
            line = line.strip('\n')
            line_str=line.split(',')
            #Array[0]=line_str[0]
            # if " "in line_str[0]:
            #     continue
            for linez in line_str[1:]:
                if linez in alien:
                    num=alien[linez]
                    Array[num]=1;
            with open('plants2.data.txt','a+')as file_object3:
                a+=1
                for i in range(69):
                    file_object3.write(str(Array[i])+' ')
                    if i == 69:
                        file_object3.write(Array[i])
                file_object3.write('\n')
                print('成功写入'+str(a)+'行')
                file_object3.close()
#将州数据存入字典中
def get_StateM():
    with open('stateabbr.txt')as file_object:
        i = 0
        for line in file_object:
            line=line.split()
            alien[line[0]]=i
            i+=1
    print(alien)
    print(i)
