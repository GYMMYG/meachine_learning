import pandas as pd
import numpy as np
import matplotlib as mp
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier  
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

#假设恶性为正例良性为反例


def acu(y_predict,y_test):   #精度 混淆矩阵 ，P，R，F1
    TP = FN = FP = TN = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_predict[i] == 1:
            TP += 1
        elif y_test[i] == 1 and y_predict[i] == 0:
            FN += 1
        elif y_test[i] == 0 and y_predict[i] == 1:
            FP += 1
        else:
            TN += 1
    #精度
    accu = (TP + TN) / len(y_predict)
    
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return accu,TP,FN,FP,TN,P,R,F1

def roc_auc(y_predict,y_test):  #roc-auc
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[0, 1], ylim=[0, 1], title='ROC',
        ylabel='TPR', xlabel='FPR')

    fpr,tpr,threshold = roc_curve(y_test, y_predict,pos_label=1)            
    ax.plot(fpr,tpr)
    plt.show()
    
    #AUC值
    Auc = roc_auc_score(y_test, y_predict)
    return Auc

def datadisc():    # 初始数据描述
    data_discrible = '''
    原始数据：
    1. Number of Instances: 699 (as of 15 July 1992)

    2. Number of Attributes: 10 plus the class attribute

    3. Attribute Information: (class attribute has been moved to last column)

    #  Attribute                     Domain
    -- -----------------------------------------
    1. Sample code number            id number
    2. Clump Thickness               1 - 10
    3. Uniformity of Cell Size       1 - 10
    4. Uniformity of Cell Shape      1 - 10
    5. Marginal Adhesion             1 - 10
    6. Single Epithelial Cell Size   1 - 10
    7. Bare Nuclei                   1 - 10
    8. Bland Chromatin               1 - 10
    9. Normal Nucleoli               1 - 10
    10. Mitoses                       1 - 10
    11. Class:                        (2 for benign, 4 for malignant)

    4. Missing attribute values: 16

    There are 16 instances in Groups 1 to 6 that contain a single missing 
    (i.e., unavailable) attribute value, now denoted by "?".  

    5. Class distribution:
    
    Benign: 458 (65.5%)
    Malignant: 241 (34.5%)
    '''
    print(data_discrible)

if __name__ == '__main__':
    #1.导入数据
    f = open('D:\桌面\\breast-cancer-wisconsin.txt','r')
    
    datadisc()  #初始数据描述

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
    #print(data01)
    data01x = np.array(data01[:,:-1])
    data01y = data01[:,-1]
    data01y = np.array([int(x) for x in data01y])  #数据类别 转换为0-1，0为良


    #留出法
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data01x, data01y, test_size = 0.2, shuffle = True)
    print('留出法划分出%d个训练集，%d个测试集'%(len(y_train),len(y_test)))

    #留出法数据用于决策树模型
    tree_clf = DecisionTreeClassifier(max_depth = 9)  # 初始化决策树模型  # 提示快捷键ctrl+Q
    tree_clf.fit(X_train, y_train)  # 训练模型
    y_predict = tree_clf.predict(X_test)  # 利用训练模型进行预测
    print('留出法构建的决策树模型：')
    print('测试集验证结果：')
    print(y_predict)

    # 决策树性能度量
    accu,TP,FN,FP,TN,P,R,F1 = acu(y_predict,y_test)
    print('精度:', accu)
    print('混淆矩阵：')
    print('TP:%d FN:%d'%(TP,FN))
    print('FP:%d TN:%d'%(FP,TN))
    print('查准率：%f\n查全率：%f\nF1:%f'%(P,R,F1))
    print()

    #k折交叉验证法 用于logistic回归
    print('交叉验证法构建的logistic模型：')
    KF=StratifiedKFold(n_splits=5) #建立5折交叉验证
    accu1 = TP1 = FN1 = FP1 = TN1 = P1 = R1 = F11 = 0
    Auc1 = 0
    for train_index,test_index in KF.split(data01x,data01y):
        X_train,X_test=data01x[train_index],data01x[test_index]
        y_train,y_test=data01y[train_index],data01y[test_index]
        # 逻辑回归模型
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        print('模型参数：')
        print(log_model.coef_)
        y_predict = log_model.predict(X_test)    #类别
        y_pred = log_model.predict_proba(X_test)   #值

        #性能度量
        accu,TP,FN,FP,TN,P,R,F1 = acu(y_predict,y_test)
        accu1 += accu
        P1 += P
        R1 += R
        F11 += F1
        Auc = roc_auc(y_pred[:,1],y_test)
        Auc1 += Auc

    accu1 /= 5
    P1 /= 5
    R1 /= 5
    F11 /= 5
    Auc1 /= 5
    print('精度:', accu1)
    print('查准率：%f\n查全率：%f\nF1:%f'%(P1,R1,F11))
    print('AUC:%f'%Auc1)    






