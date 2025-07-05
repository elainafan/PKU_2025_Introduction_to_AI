import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 2e-3  # 学习率
wd = 1e-4  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return X.dot(weight)+bias
    raise NotImplementedError

def sigmoid(x):
    index_positive=np.nonzero(x>0)
    index_negative=np.nonzero(x<0)
    temp_x=np.zeros_like(x)
    temp_x[index_positive]=1/(np.exp(-x[index_positive])+1)
    temp_x[index_negative]=np.exp(x[index_negative])/(np.exp(x[index_negative])+1) 
    return temp_x
    # 改sigmoid函数，进行分类，防止exp的正数爆
def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    haty=sigmoid(predict(X,weight,bias))
    loss=np.average(np.log(sigmoid(predict(X,weight,bias)*Y)))
    d_weight=-lr*np.dot(X.T,((1-sigmoid(predict(X,weight,bias)*Y)))*Y)+2*wd*lr*weight
    d_bias=-lr*np.average((1-sigmoid(predict(X,weight,bias))*Y)*Y)
    weight-=d_weight
    bias-=d_bias
    return (haty,loss,weight,bias)
    raise NotImplementedError
