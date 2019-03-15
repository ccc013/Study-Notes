#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/21 14:38
@Author  : cai

实现神经网络算法
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

# 定义Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义Sigmoid函数的梯度计算
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 前向传播计算，网络层总共三层，分别是输入，隐层和输出层
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    # 隐层的输入
    z2 = a1 * theta1.T
    # 隐层的输出
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    # 输出层的输入
    z3 = a2 * theta2.T
    # 输出层的输出，也就是预测结果
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

# 反向传播
def backprop(params, input_size, hidden_size, num_labels, X, y, lambdas):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 对每一层，调整权值参数矩阵的尺寸
    # 隐层的权值矩阵, (25, 401)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1))))
    # 输出层的权值矩阵, (10, 26)
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1))))

    # 前向传播计算
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 计算误差
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m
    # 添加正则化
    J += (float(lambdas) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # 下面是反向传播计算过程
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))   # (1, 25)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m
    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * lambdas) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * lambdas) / m
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex3data1.mat')
# 载入数据
data = loadmat(dataPath)
# print(data)
X = data['X']
y = data['y']
print(X.shape, y.shape)

# 利用sklearn包的函数将标签从一个整数变成一个长度为k的向量，k是类别的总数
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
# print(y_onehot.shape)
# print(y[0], y_onehot[0, :])

# 初始参数
input_size = 400
hidden_size = 25
num_labels = 10
lambdas = 1

# 随机初始化权值参数
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

# 测试数据
# m = X.shape[0]
# X = np.matrix(X)
# y = np.matrix(y)

# unravel the parameter array into parameter matrices for each layer
# theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
# theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
#
# print(theta1.shape, theta2.shape)
#
# a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# print(a1.shape, z2.shape, a2.shape, z3.shape, h.shape)

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, lambdas)
print('cost = ', J)
print(grad.shape)

# 最小化损失函数
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, lambdas),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)

# 进行预测
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))

