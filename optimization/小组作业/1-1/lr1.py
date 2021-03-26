import pandas as pd
import numpy as np
from sklearn import preprocessing

class Data_solve:
    '''
    处理数据：标准化 或 归一化
    '''
    def __init__(self):
        self.data = []
        self.data01 = []    #标准化
        self.data0_1 = []    #归一化
        self.ranges0_1 = []
        self.mins0_1 = []
        self.ave01 = []
        self.var01 = []

    def fit(self,data):
        self.data = data
        self.norm_data()
        self.stadard_data()
    
    def norm_data(self):
        data = self.data
        mins = data.min(0)      #0为列，1为行 
        maxs = data.max(0)      
        ranges = maxs - mins    
        normData = np.zeros(np.shape(data))     #用于装归一化后的数据
        row = data.shape[0]                     #行数
        normData = data - np.tile(mins,(row,1)) #np.title扩大数组 参数为 数组 ，行扩大倍数，列。。。
        normData = normData / np.tile(ranges,(row,1))   

        self.data0_1 = normData                 #归一数组，范围，最小值
        self.mins0_1 = np.array(mins) 
        self.ranges0_1 = np.array(ranges)

    def stadard_data(self):
        data = self.data
        self.ave01 = data.mean(axis=0)      #标准化后的数据，均值方差
        self.var01 = data.mean(axis=0)
        self.data01 = preprocessing.scale(data)

class Linear_regression():
    '''
    线性回归
    '''
    def __init__(self):
        self.data = []
        self.X = []
        self.Y = []
        self.b = []   #初始系数 
        self.b1 = []   #正规方程系数
        self.b2 = []   #后退法系数
        self.b3 = []   #固定步长系数
        self.bavr = []  #平均值
        self.sub = []  #偏移量
        self.mul = []   #缩小量
    
    def fit(self,data,mul,sub):
        X = np.mat(data[:,:-1])
        X = np.append(X,[[1] for _ in range(len(X))],axis = 1)  #加上常数项
        Y = np.mat(data[:,-1])      #只切一列时为行!!!
        Y = Y.T
        row,column = X.shape    
        b = np.mat([0 for _ in range(column)]).T    #设定起始点列向量
        self.data = data
        self.X = X
        self.Y = Y
        self.b = b
        self.sub = sub
        self.mul = mul
        self.b1 = self.lre()    #正规方程
        self.b2 = self.lrb()    #后退
        self.b3 = self.lrf()    #固定步长
        self.bavr.append(self.b1)
        self.bavr.append(self.b2)
        self.bavr.append(self.b3)
        self.bavr = np.array(self.bavr).mean(axis = 0)  #系数平均

    def lre(self):              #正规方程法解系数
        #θ = (xT*x)^-1*xT*Y
        result = (self.X.T*self.X).I*self.X.T*self.Y
        print('正规方程法求得系数为：',end = ' ')
        print(result.T)
        return np.array(result.T)

    
    def fun(self,b):                       #当前函数值
        return self.X*b

    def f_j(self,b):                     #当前误差
        Y1 = self.fun(b)                       #Y1：当前函数值   
        return 0.5 * self.sqrl(self.Y - Y1)
    
    def sqrl(self,X):                         #矩阵二范数的平方
        return (np.mat((np.array(X))**2)).sum()

    def flo(self,t,b):      #计算更新点
        db = self.fl(b)        #一阶梯度阵
        return b - t * db    #返回更新后的值

    def fl(self,b):      #计算一阶梯度值
        dj = self.fun(b) - self.Y
        db = self.X.T * dj   
        return db
    
    def lrb(self):              #后退法
        b = self.b

        for i in range(300):                     #调试发现300为 终止迭代步数        
            t = 1                         #设定初始步长
            b1 = self.flo(t,b)
            while self.f_j(b1) > (self.f_j(b) - 0.5 * t * self.sqrl(self.fl(b).T)):   #判断后退是否合理
                t = 0.5 * t                                  #更新后退长度
                b1 = self.flo(t,b)
            
            b = b1
        
        print('后退法求得系数为：',end = ' ')
        print(b.T)
        return np.array(b.T)

    def lrf(self):              #固定步长法
        b = self.b

        for i in range(3000):                     #调试发现3000为 终止迭代步数        
            t = 0.001                         #设定初始步长
            b = self.flo(t,b)

        print('固定步长法求得系数为：',end = ' ')
        print(b.T)
        return np.array(b.T)

    def predict(self):
        #测试数据  
        while True:                  
            print('输入待预测数据参数：(输入-1结束)')
            data_test = [float(x) for x in input().split()]

            if data_test[0] == -1:       #  输入-1结束
                break

            print(data_test,end = ' ')
            print('三种方法的平均预测结果为：')
            self.pre(np.array(data_test))    #预测
            
            print()

    def pre(self,data):   #预测
        mins = self.sub
        ranges = self.mul
        factors = np.mat(self.bavr).T
        normData = (data - mins[:-1]) / ranges[:-1]     #归一化 

        result = np.mat(normData) * factors[:-1,:]            #计算结果  !！注意常数项
        result = (result + factors[-1,-1]) * ranges[-1] + mins[-1]     #去归一化
        print(result[0,0])


if __name__ == '__main__':
    #数据处理为[X Y]型：
    df = pd.read_excel('D:\桌面\最优化数据1.xlsx',sheet_name=0)  #读入数据

    data_column = df.columns.values[1:]

    dataori = np.array(df.values)       #获取全部数据
    data = dataori[: ,1:]                #去除第一列索引

    #   构造归一化 或标准化数据，这里采用归一化 
    dt = Data_solve()
    dt.fit(data)

    #构造线性回归模型
    lr = Linear_regression()
    lr.fit(dt.data0_1,np.array(dt.ranges0_1),np.array(dt.mins0_1))  #归一化数据
    lr.predict()       #预测结果


    