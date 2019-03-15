# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:50:06 2016

@author: cai
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


# 计算损失函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# 梯度下降算法
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            # 计算误差对权值的偏导数
            term = np.multiply(error, X[:, j])
            # 更新权值
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex1data2.txt')
data = pd.read_csv(dataPath, header=None, names=['Size', 'Bedrooms', 'Price'])
# print(data.head())
# print(data.describe())

# 对数据做归一化
data = (data - data.mean()) / data.std()
# print(data.head())

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# 从数据帧转换成numpy的矩阵格式
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.zeros((1, cols-1)))

# 初始化学习率和迭代次数
alpha = 0.01
iters = 1000
# 实现线性回归算法
g, cost = gradientDescent(X, y, theta, alpha, iters)
# 获取模型的误差
print('cost = ', computeCost(X, y, g))
# 可视化误差和训练迭代次数的曲线图
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()





