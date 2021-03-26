import pandas as pd
import numpy as np

'''
mpg作为因变量
去除车名
共七个自变量（'cylinders', 'displacement'  , 'horsepower',
 'weight' ,'acceleration',  'model_year' , 'origin'）
去除有缺失数据行
'''

def norm(data):         #数据归一化  0-1之间
    mins = data.min(0)      #0为列，1为行 
    maxs = data.max(0)      
    ranges = maxs - mins    
    normData = np.zeros(np.shape(data))     #用于装归一化后的数据
    row = data.shape[0]                     #行数
    normData = data - np.tile(mins,(row,1)) #np.title扩大数组 参数为 数组 ，行扩大倍数，列。。。
    normData = normData / np.tile(ranges,(row,1))   
    return normData,np.array(ranges),np.array(mins)             #返回归一数组，范围，最小值



def lre(data):              #正规方程法解系数
    #θ = (xT*x)^-1*xT*Y
    X = np.mat(data[:,:-1])
    X = np.append(X,[[1] for _ in range(len(X))],axis = 1)  #加上常数项
    Y = np.mat(data[:,-1])      #只切一列时为行!!!
    # print(X)
    # print(Y.T)
    Y = Y.T
    result = (X.T*X).I*X.T*Y
    print('正规方程法求得系数为：',end = ' ')
    print(result.T)
    return np.array(result.T)


def fun(X,b):                       #当前函数值
    return X*b


def f_j(Y,X,b):                     #当前误差
    Y1 = fun(X,b)                       #Y1：当前函数值   
    return 0.5 * sqrl(Y - Y1)


def sqrl(X):                         #矩阵二范数的平方
    return (np.mat((np.array(X))**2)).sum()


def flo(Y,X,b,t):      #计算更新点
    db = fl(Y,X,b)        #一阶梯度阵
    #print(db)
    return b - t * db    #返回更新后的值


def fl(Y,X,b):      #计算一阶梯度值
    dj = fun(X,b) - Y
    db = X.T * dj   

    return db
    

def datatr(data):      #数据处理
    X = np.mat(data[:,:-1])
    X = np.append(X,[[1] for _ in range(len(X))],axis = 1)  #加上常数项
    Y = np.mat(data[:,-1])      #只切一列时为行!!!
    Y = Y.T
    row,column = X.shape    

    b = np.mat([0.5 for _ in range(column)]).T    #设定起始点列向量
    return X,Y,b


def lrb(data):              #后退法
    X,Y,b = datatr(data)   #数据 处理
    t = 1               #设定初始步长
    for i in range(20000):                     #调试发现20000为 大致终止迭代步数,但并不完全相同    
        yj = f_j(Y,X,b)                  #当前误差
        # print(yj)
        #t = 1               #设定初始步长
        b1 = flo(Y,X,b,t)
        while f_j(Y,X,b1) > (f_j(Y,X,b) - 0.5 * t * sqrl(fl(Y,X,b).T)):   #判断后退是否合理
            t = 0.5 * t                                  #更新后退长度
            b1 = flo(Y,X,b,t)
        
        b = b1
       # print(b)
    print('后退法求得系数为：',end = ' ')
    print(b.T)
    return np.array(b.T)


def lrf(data):              #固定步长法
    X,Y,b = datatr(data)   #数据 处理
    
    for i in range(20000):                     #调试发现20000为 终止迭代步数        
        #yj = f_j(Y,X,b)                  #当前误差
        # print(yj)
        t = 0.001                         #设定初始步长
        b = flo(Y,X,b,t)
        
    print('固定步长法求得系数为：',end = ' ')
    print(b.T)
    return np.array(b.T)


def pre(data,factors,ranges,mins):   #  预测结果
    normData = np.zeros(np.shape(data))
    normData = (data - mins[:-1]) / ranges[:-1]     #归一化 

    result = np.mat(normData) * factors[:-1,:]            #计算结果  !！注意常数项
    result = (result + factors[-1,-1]) * ranges[-1] + mins[-1]     #去归一化
    print(result[0,0])


def Linear_regression(data):
        #数据归一化
        data01,ranges,mins = norm(data)   
        # print(data01)
        
        #三种方法系数
        factors = []
        factors.append(lre(data01))            #正规方程法求系数  
        factors.append(lrb(data01))           #后退法求系数
        factors.append(lrf(data01))          #固定步长求系数
        factors_ave = np.array(factors).mean(axis = 0)  #系数平均
        
        #测试数据  
        while True:                  
            print('输入待预测数据（7个参数）：(-1结束)')
            data_test = [float(x) for x in input().split()]

            if data_test[0] == -1:       #  输入-1结束
                break

            print(data_test,end = ' ')
            print('三种方法的平均预测结果为：')
            pre(np.array(data_test),np.mat(factors_ave).T,ranges,mins)    #预测
            
            print()


if __name__ == '__main__':

    #数据处理为[X Y]型：
    f = open('D:\桌面\最优化数据2.txt','r')
    datas = f.readlines()
    data = []
    for dataline in datas:
        ndata = [x for x in dataline.split()]           #逐行读取数据
        ndata = ndata[:8]                               #去除车型
        # print(ndata)

        if '?' in ndata:                                 #去除缺失数据行
            continue            

        ndata = list(map(float,ndata))                    #转换为数值
        ndata = ndata[1:] + [ndata[0]]                   #将Y变量放在最后
        data.append(ndata)
   # print(data)
    f.close() 
    data_name = ['cylinders', 'displacement'  , 'horsepower', 'weight' ,'acceleration',  'model_year' , 'origin' , 'mpg' ] #变量名
    data = np.array(data)
    
    #线性回归
    Linear_regression(data)
    