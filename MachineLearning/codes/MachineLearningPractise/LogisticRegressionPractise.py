# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:28:06 2016

@author: cai
实现逻辑回归算法
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy.optimize as opt

# 定义Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义 cost函数
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = X * theta.T
    first = np.multiply(-y, np.log(sigmoid(h)))
    second = np.multiply(1-y, np.log(1 - sigmoid(h)))
    return np.sum(first - second) / (len(X))

# 梯度下降算法的实现, 输出梯度对权值的偏导数
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

# 预测结果
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x  in probability]


dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex2data1.txt')
data = pd.read_csv(dataPath,header=None,names=['Exam 1', 'Exam 2', 'Admitted'])
# 查看数据集
# print(data.head())
# print(data.describe())

# 分成正负两个数据集
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
# 可视化数据集
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='No Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

# 可视化 sigmoid函数
# nums = np.arange(-10, 10, step=1)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(nums, sigmoid(nums), 'r')
# plt.show()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# 从数据帧转换成numpy的矩阵格式
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.zeros((1, cols-1))
print(X.shape, theta.shape, y.shape)

costs = cost(theta, X, y)
print('cost = ', costs)

# 使用scipy库中的优化函数,得到训练好的权值
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
# print(cost(result[0], X, y))

# 预测结果，统计分类准确率
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))



