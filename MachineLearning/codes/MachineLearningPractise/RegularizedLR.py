# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:03:06 2016

@author: cai

实现正则化的逻辑回归算法
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
def gradientReg(theta, X, y, lambdas):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((lambdas / len(X)) * theta[:, i])

    return grad

# 预测结果
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex2data2.txt')
data = pd.read_csv(dataPath, header=None, names=['Test 1', 'Test 2', 'Accepted'])
# 查看数据集
print(data.head())
# print(data.describe())

# 分成正负两个数据集
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]
# 可视化数据集
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='No Accepted')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()

# 数据预处理
degree = 5
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)
# 构造多项式
for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)
print(data.head())

# 数据准备
cols = data.shape[1]
X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]

# 从数据帧转换成numpy的矩阵格式
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros((1, cols-1))
print(X.shape, theta.shape, y.shape)

lambdas = 1

print(costReg(theta, X, y, lambdas))

# costs = cost(theta, X, y)
# print('cost = ', costs)

# 使用scipy库中的优化函数
result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, lambdas))
# print(cost(result[0], X, y))
# print(result)

# 预测结果，统计分类准确率
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))



