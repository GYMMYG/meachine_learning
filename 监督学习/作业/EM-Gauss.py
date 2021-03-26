import numpy as np



def load_data(mean,cov,number):
    data = np.random.multivariate_normal(mean,cov,number)
    #print(data)
    return data

#EM算法求解Gauss
def EM_Gauss(data,k):
    #初始化参数
    A = np.array([1.0/k for _ in range(k)])[:,np.newaxis]
    P = np.hstack([np.mat([1.0/k for _ in range(k)]).T ,np.mat([1.0/k for _ in range(k)]).T ])
    E = np.ones((k,2,2),dtype=float) / k
    n = len(data)  #样本数
   
    #迭代至收敛
    t = 0
    while t == 0 or (not judge()):    #收敛判断
        #1.E步

        #2.M步

        
# 计算高斯函数
def Gaussian(data,mean,cov):
    dim = np.shape(cov)[0]   # 计算维度
    covdet = np.linalg.det(cov) # 计算|cov|
    covinv = np.linalg.inv(cov) # 计算cov的逆
    if covdet==0:              # 以防行列式为0
        covdet = np.linalg.det(cov+np.eye(dim)*0.01)
        covinv = np.linalg.inv(cov+np.eye(dim)*0.01)
    m = data - mean
    z = -0.5 * np.dot(np.dot(m, covinv),m)    # 计算exp()里的值
    return 1.0/(np.power(np.power(2*np.pi,dim)*abs(covdet),0.5))*np.exp(z)  # 返回概率密度值



if __name__ == "__main__":
    #生成900个数据
    datas = load_data([3,1],[[1,-0.5],[-0.5,1]],300)
    datas = np.vstack([datas,load_data([8,10],[[2,0.8],[0.8,2]],300)]) 
    datas = np.vstack([datas,load_data([12,2],[[1,0],[0,1]],300)]) 

    #2-5个高斯混合成分

    #A-生成概率，P-均值，E-协方差矩阵
    A2,P2,E2 = EM_Gauss(datas,2)
    A3,P3,E3 = EM_Gauss(datas,3)
    A4,P4,E4 = EM_Gauss(datas,4)
    A5,P5,E5 = EM_Gauss(datas,5)
    


        
