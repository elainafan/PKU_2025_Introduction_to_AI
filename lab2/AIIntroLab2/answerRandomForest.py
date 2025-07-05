from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 25    # 树的数量
ratio_data = 0.8   # 采样的数据比例
ratio_feat = 0.4 # 采样的特征比例
hyperparams = {
    "depth":25, 
    "purity_bound":1.2,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    trees=[]
    for i in range(num_tree):
        random_data=np.random.rand(np.shape(X)[0])
        temp_x=[]
        temp_y=[]
        for j in range(np.shape(X)[0]):
            if random_data[j]<=ratio_data:
                temp_x.append(X[j])
                temp_y.append(Y[j])
        temp_feat=[]
        random_feat=np.random.rand(np.shape(X)[1])
        for j in range(np.shape(X)[1]):
            if random_feat[j]<=ratio_feat:
                temp_feat.append(j)
        X_0=np.array(temp_x)
        Y_0=np.array(temp_y)
        trees.append(buildTree(X_0,Y_0,temp_feat,depth=hyperparams["depth"],purity_bound=hyperparams["purity_bound"],gainfunc=hyperparams["gainfunc"],prefixstr=""))
    return trees
    # 提示：整体流程包括样本扰动、属性扰动和预测输出

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
