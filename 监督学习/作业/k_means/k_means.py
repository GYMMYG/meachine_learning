import numpy as np
from numpy import *

def load_data(file_name):
    f = open(file_name)
    datas = []
    t = 0
    for line in f.readlines():
        datas.append(line.strip().split())
    return datas        

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

def randCent(dataset,k): #随机法生成中心点矩阵
    n = shape(dataset)[1]   #n个属性
    centset = mat(zeros((k,n)))   #初始化中心点矩阵
    for i in range(n):
        mini = min(dataset[:,i])
        maxi = max(dataset[:,i])
        rangei = float(maxi - mini)
        centset[:,i] = mini + rangei*random.rand(k,1)
    return centset

def k_means(dataset,k, distMeans =distEclud, createCent = randCent):
    centset = createCent(dataset,k)  #存放初始化中心点
    m = shape(dataset)[0]
    type_dis = mat(zeros((m,2)))    #存放每一个数据的中心点及距中心点距离
    centerchange = True  #判断是否收敛

    #算法迭代过程
    while centerchange:
        centerchange = False
        #对每一个点计算其与中心距离，找出最小中心距离
        for i in range(m):
            minidx = -1
            mindist = inf
            for j in range(k):
                centdist = distMeans(centset[j,:].T,dataset[i,:].T)
                if centdist < mindist:
                    mindist = centdist
                    minidx = j
            if type_dis[i,0] != minidx:
                centerchange = True
            type_dis[i] = minidx,mindist**2
        
        #更新中心点   #即使centerchange为false仍然要更新一次，因为中心点可能变化
        for i in range(k):
            #计算此时各类点
            dataset = np.array(dataset)
            centdata = []
            for j in range(m):
                if type_dis[j,0] == i:
                    centdata.append(dataset[j])#直接append会变成三维
        #     x = type_dis[:,0]
        #    # print(x)
        #     y = x == i
        #     y = np.tile(y,(1,shape(dataset)[1]))
        #     #print(y)
        #     centdata = dataset[y]
        #     k = size(centdata) //2
        #     centdata = centdata.reshape(k,2)
        #    # print(centdata)
        #     #计算更新点坐标
        #     # print(centdata)
            #print(np.array(centdata))
            print(np.array(centdata))
            centset[i] = mean(np.array(centdata),axis =0)
            
        print(centset)
        
    
    return centset,type_dis
        





if __name__ == '__main__':
    #data processing
    file_name = 'D:\桌面\\k_means.txt'
    datas = load_data(file_name)
    data = datas[1:]
    for i in range(len(data)):
        data[i] = list(map(float,data[i]))
    data = np.mat(data)
    data_type = datas[0]
    #k_means
    k = 4
   # k_means(data,k)
    type_means,type_dis = k_means(data,k)
    
    # print(type_means)
    # print(type_dis)

#画图过程待补充

    

    

    
