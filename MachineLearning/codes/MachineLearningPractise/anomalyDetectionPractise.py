#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/22 15:37
@Author  : cai

使用高斯模型实现一个异常检测的算法
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from scipy import stats

# 返回数据的均值和方差
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

#
def select_threshold(pval, yval):
    best_epsiln = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        # 预测为真，实际为真
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        # 预测为真，但是实际为假
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        # 预测为假，实际为真
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsiln = epsilon

    return best_epsiln, best_f1

dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex8data1.mat')
data = loadmat(dataPath)
X = data['X']
print(X.shape)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.set_title('original data')
# plt.show()

mu, sigma = estimate_gaussian(X)
# print('mean={0}\nvar={1}'.format(mu, sigma))
Xval = data['Xval']
yval = data['yval']
# print(Xval.shape, yval.shape)
# dist = stats.norm(mu[0], sigma[0])
# print(dist.pdf(X[:, 0])[0:50])
# 计算概率密度
p = np.zeros((X.shape[0], X.shape[1]))
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])
print(p.shape)
# 计算验证集的概率密度
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])

epsilon, f1 = select_threshold(pval, yval)
print('epsilon=', epsilon)
print('f1=', f1)

# 检测出异常的数据
outliers = np.where(p < epsilon)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
ax.set_title('detection result')
plt.show()


