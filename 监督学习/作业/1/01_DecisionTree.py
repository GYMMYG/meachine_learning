# encoding=utf-8

# 随机构造数据集
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2)

# 构造训练集和测试集
X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]
print(y_test)
# 构造决策树分类模型
from sklearn.tree import DecisionTreeClassifier  # 导入决策树所在的包
tree_clf = DecisionTreeClassifier(max_depth=3)  # 初始化决策树模型  # 提示快捷键ctrl+Q
tree_clf.fit(X_train, y_train)  # 训练模型
y_predict = tree_clf.predict(X_test)  # 利用训练模型进行预测
print(y_predict)

# 评价方法性能：在测试数据上的分类准确率
accu = tree_clf.score(X_test, y_test)
print('Test accuracy:', accu)

# 画图
from sklearn.tree import export_graphviz
# 使用graphviz打开fig.dot文件查看图片
export_graphviz(
    tree_clf,
    out_file=".//fig.dot",
    rounded=True,
    filled=True
)
