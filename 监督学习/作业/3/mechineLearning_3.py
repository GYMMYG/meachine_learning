import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import exp

def fun(X,p,Y):          #计算梯度
    return  X.T * (p - Y)

def fp(b,X):              #计算p
    bt = b.T
    p = []

    for i in range(len(X)):
        y = -1 * bt * X[i].T
        p.append(1 / (1 + exp(y)))
    return np.mat(p).T

def datatr(datax,datay):            #处理数据
    fac_len = len(datax[0])
    X = np.mat(datax)
    X = np.append(X,[[1] for _ in range(len(X))],axis = 1)  #加上常数项
    Y = (np.mat(datay)).T
    b = np.mat([0 for _ in range(fac_len + 1)])
    print(b)
    b = b.T
    return X,Y,b

def gd(datax,datay):            #梯度下降算法
    X,Y,b = datatr(datax,datay)
    t = 0.01
    for i in range(1000):             #1000次之后发现变化较小
        p = fp(b,X)
        b = b - t * fun(X,p,Y)

    b = b.T
    print('梯度下降法(1000步迭代)b：')
    print(b)

def dia(p):              #对角矩阵
    n,m = np.shape(p)
    normData = np.zeros((m,m))
    for i in range(m):
        normData[i,i] = p[0,i] * (1 - p[0,i])
    return normData

def nt(datax,datay):           #牛顿法
    X,Y,b = datatr(datax,datay)
    for i in range(10):            #十步之内就可以得到最终解
        p = fp(b,X)
        gk = fun(X,p,Y)
        S = dia(p.T)
        H = X.T * S * X
        db = -H.I * gk
        b = b + db
    print('牛顿法(10步迭代)b:')
    print(b.T)


if __name__ == '__main__':
    #1.导入数据
    f = open('D:\桌面\\breast-cancer-wisconsin.txt','r')

    #数据处理  1.去除id  2.去除缺失值  3.标准化数据
    datas = f.readlines()
    data = []
    for dataline in datas:
        ndata = [x for x in dataline.split(',')]     #以逗号分隔数据
        
        #1.去除id
        ndata = ndata[1:]                             
        #2.去除缺失数据行
        if '?' in ndata:                                 
            continue  

        ndata = list(map(float,ndata))                    #转换为数值,浮点数方便运算

        data.append(ndata)          
    
    print('数据处理过后有%d个属性，%d个样例'%(len(data[1])-1,len(data)))
    print()
    f.close()

    data = np.array(data)
    # print(data)

    #3.数据标准化
    data01 = preprocessing.scale(data)
    data01x = np.array(data01[:,:-1])
    data01y = data01[:,-1]
    data01y = np.array([int(x) for x in data01y])  #数据类别 转换为0-1，0为良
    
    #梯度下降法
    gd(data01x,data01y)
    
    #牛顿法
    nt(data01x,data01y)