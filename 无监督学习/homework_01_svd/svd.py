import numpy as np
from PIL import Image


def imgCompress(channel, percent):
    U, sigma, V_T = np.linalg.svd(channel)
    m = U.shape[0]
    n = V_T.shape[0]
    reChannel = np.zeros((m, n))

    for k in range(len(sigma)):
        reChannel = reChannel + sigma[k] * np.dot(U[:, k].reshape(m, 1), V_T[k, :].reshape(1, n))
        if float(k) / len(sigma) > percent:
            reChannel[reChannel < 0] = 0
            reChannel[reChannel > 255] = 255
            break

    return np.rint(reChannel).astype("uint8")


oriImage = Image.open(r'D:\桌面\奇异值.jpg', 'r')
imgArray = np.array(oriImage)
print(imgArray)
# R = imgArray[:, :, 0]
# G = imgArray[:, :, 1]
# B = imgArray[:, :, 2]
# # A = imgArray[:, :, 3]
#
# for p in [0.01, 0.02,  0.05, 0.1, 0.2, 0.3]:
#     reR = imgCompress(R, p)
#     reG = imgCompress(G, p)
#     reB = imgCompress(B, p)
#     # reA = imgCompress(A, p)
#     reI = np.stack((reR, reG, reB), 2)
#
#     Image.fromarray(reI).save("{}".format(p) + "img.png")





import numpy as np #科学计算库
import matplotlib.pyplot as plt # 绘图库
import cv2 # 图形处理库，这里用于通道的合并


def svdCompression(imgFile, K):
    # 读取图片
    img = plt.imread(imgFile)
    # 提取三个通道
    imgR, imgG, imgB = img[:,:,0], img[:,:,1], img[:,:,2]
    # 计算三个通道提取前K个奇异值的压缩矩阵
    R1,contriR = mySVDcompression(imgR,K)
    G1,contriG = mySVDcompression(imgG,K)
    B1,contriB = mySVDcompression(imgB,K)
    #合并三个通道
    img1 = cv2.merge([R1,G1,B1]) 
    img1 = np.rint(img1).astype('uint8')
    return img1,contriR,contriG, contriB


def mySVDcompression(img_matrix,K):
    m,n = img_matrix.shape
    A = np.mat(img_matrix, dtype = float)
    # 计算AA'特征值和向量
    lambda1, U = np.linalg.eig(A.dot(A.T))
    # 计算累积贡献率
    contri = contribution(lambda1)
    S = np.ones((m, n))
    S[:m,:m] = np.diag(np.sqrt(lambda1))
    US = U.dot(S)
    V = US.I.dot(A)
    S[K:,:] = 0
    S[:,K:] = 0
    return U.dot(S).dot(V),contri

def contribution(L):
    a = []
    b = 0
    for ii in L:
        b = b+ii/L.sum()
        a.append(b)
    return a
    
    
if __name__ == "__main__":
    for K in range(10,20,20):
       # 这里图像可以设置为其他图像
        Img,contriR,contriG, contriB= svdCompression('butterfly.bmp', K)
        plt.subplot(2,3,(K-10)/20+1)
        plt.imshow(Img)
        plt.title(('using %d singular values'%K))
    plt.figure()
    plt.plot(contriR)
    plt.plot(contriG)
    plt.plot(contriB)
    plt.legend(['Tunnel R','Tunnel G','Tunnel B'])
    plt.title('Variation of Singular Value Contribution Rate')
    plt.xlabel('Number of Singular Values')
    plt.ylabel('Contribution Rate')
————————————————
版权声明：本文为CSDN博主「UESTC Like_czw」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40527086/article/details/88925161