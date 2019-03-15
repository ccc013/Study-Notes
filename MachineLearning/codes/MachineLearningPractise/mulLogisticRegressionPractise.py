#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/19 21:35
@Author  : cai

实现多类的逻辑回归算法
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.io import loadmat

# 定义Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义 cost函数
def costReg(theta, X, y, lambdas):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = X * theta.T
    first = np.multiply(-y, np.log(sigmoid(h)))
    second = np.multiply((1-y), np.log(1 - sigmoid(h)))
    reg = (lambdas / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

# 梯度下降算法的实现, 输出梯度对权值的偏导数
def gradient(theta, X, y, lambdas):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    # 计算误差
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((lambdas / len(X)) * theta)

    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()

# 实现一对多的分类方法
def one_vs_all(X, y, num_labels, lambdas):
    rows = X.shape[0]
    params = X.shape[1]

    # 每个分类器有一个 k * (n+1)大小的权值数组
    all_theta = np.zeros((num_labels, params + 1))

    # 增加一列，这是用于偏置值
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # 标签的索引从1开始
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # 最小化损失函数
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, lambdas), method='TNC', jac=gradient)
        all_theta[i-1, :] = fmin.x

    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # 增加一列，这是用于偏置值
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # 对每个训练样本计算其类的概率值
    h = sigmoid(X * all_theta.T)

    # 获取最大概率值的数组索引
    h_argmax = np.argmax(h, axis=1)
    # 数组是从0开始索引，而标签值是从1开始，所以需要加1
    h_argmax = h_argmax + 1

    return h_argmax

dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex3data1.mat')
# 载入数据
data = loadmat(dataPath)
print(data)
print(data['X'].shape, data['y'].shape)

# print(np.unique(data['y']))
# 测试
# rows = data['X'].shape[0]
# params = data['X'].shape[1]
#
# all_theta = np.zeros((10, params + 1))
#
# X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
#
# theta = np.zeros(params + 1)
#
# y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
# y_0 = np.reshape(y_0, (rows, 1))
# print(X.shape, y_0.shape, theta.shape, all_theta.shape)

all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta)

# 计算分类准确率
y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))

