# -*- encoding: utf8 -*-
from math import log
'''
代码错误：
1.缺少当最后剩下空集和数据集在剩余标签上取值相同时的处理
解决办法：加一个终止条件

2.缺少对训练集在某子树下数据取不到标签上的所有值的处理
解决办法：验证时当出现训练集中不存在的值时，取父节点的类别，可以在字典加父类(每一个子树中的每一个标签)本身的值
'''

class DecisionTree:
    """
    ID3算法实现的决策树
    """

    def __init__(self):

        self.__feature_labels = dict()
        self.__tree = dict()
    
    def is_same(self,data_set):          #判断data是否已经完全一致
        # for ite in datas: 
        #     del(ite[-1])
        total_len = len(data_set)
        for i in range(total_len - 1) :
            if data_set[i][:-1] != data_set[i + 1][:-1]:
                return False       
        return True

    def sel_feature(self,data_set):                 #计算数据中最大类别
        total_len = len(data_set)
        label_counts = {}

        for item in data_set:
            label_counts[item[-1]] = label_counts.get(item[-1], 0) + 1
        
        res = list(sorted(zip(label_counts.values(),label_counts.keys())))
        return res[-1][-1]

    def cul_entropy(self, data_set):
        """
        计算信息熵
        :param data_set:
        :return: float ent
        """
        total_len = len(data_set)
        label_counts = {}

        for item in data_set:
            label_counts[item[-1]] = label_counts.get(item[-1], 0) + 1

        ent = 0
        for key in label_counts:
            label_prob = float(label_counts[key]) / total_len
            ent -= label_prob * log(label_prob, 2)

        return ent

    def cul_information_gain_label(self, before_ent, data_set, axis):
        """
        计算信息增益
        :param before_ent:
        :param data_set:
        :param axis:
        :return: float information gain
        """
        total_len = len(data_set)
        feature_values = [item[axis] for item in data_set]
        unique_feature_values = set(feature_values)

        label_ent = 0
        for feature_value in unique_feature_values:
            label_data_set = self.split_data_lable(data_set, axis, feature_value)
            label_ent += float(len(label_data_set)) / total_len * self.cul_entropy(label_data_set)

        return before_ent - label_ent

    def get_best_feature(self, data_set):
        """
        获得信息增益最大的一个特征
        :param data_set:
        :return: feature axis
        """
        features_len = len(data_set[0][:-1])
        before_ent = self.cul_entropy(data_set)

        label_gain = {}

        cnt = 0
        for axis in range(features_len):
            label_gain[axis] = self.cul_information_gain_label(before_ent, data_set, axis)

            if label_gain[axis] > label_gain[cnt]:
                cnt = axis
        # return max(label_gain.iterkeys(), key=lambda k: label_gain[k])
        return cnt

    def split_data_lable(self, data_set, axis, feature_value):
        label_data_set = []
        for item in data_set:
            if item[axis] == feature_value:
                item_vec = item[:axis]
                item_vec.extend(item[axis + 1:])
                label_data_set.append(item_vec)

        return label_data_set

    def create_tree_id3(self, data_set, labels):
        class_list = [item[-1] for item in data_set]

        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]

        if (not len(labels)) or self.is_same(data_set):    #对应的第二个结束条件
            return self.sel_feature(data_set) 
        
        best_feature = self.get_best_feature(data_set)
        best_feature_label = labels[best_feature]
        del(labels[best_feature])
        my_tree = {best_feature_label: {}}

        my_tree[best_feature_label]['self'] =  self.sel_feature(data_set)          #记录每一个子树中分类标签本身的类别（由所有所属实例的最大类别 决定）

        feature_values = [item[best_feature] for item in data_set]
        unique_feature_values = set(feature_values)
        for feature_value in unique_feature_values:
            sub_labels = labels[:]
            my_tree[best_feature_label][feature_value] = self.create_tree_id3(self.split_data_lable(data_set, best_feature, feature_value), sub_labels)

        return my_tree

    def fit(self, data_set, labels):
        for axis in range(len(labels)):
            self.__feature_labels[labels[axis]] = axis

        self.__tree = self.create_tree_id3(data_set, labels)
        print ("tree:", self.__tree)

    def predict_tree(self, tree, product):
        for key in tree:
            feature_value = product[self.__feature_labels[key]]

            if  feature_value not in tree[key].keys():                    #当不存在时返回父节点的类别
                return tree[key]['self']

            if isinstance(tree[key][feature_value], dict):
                return self.predict_tree(tree[key][feature_value], product)
            else:
                return tree[key][feature_value]

    def predict(self, products):
        return [self.predict_tree(self.__tree, product) for product in products]


def load_data():
    labels = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play']
    data_set = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ]
    return data_set, labels


if __name__ == '__main__':
    data_set, labels = load_data()
    dt = DecisionTree()
    #print(labels)
    dt.fit(data_set, labels)
    
    products = [
                ['Rain', 'Mild', 'High', 'Strong'],
                ['Overcast', 'Hot', 'High', 'Strong'],
                ]

    probs = dt.predict(products)
    print ("probs:", probs)
   # print(labels)
