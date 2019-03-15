#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/22 14:58
@Author  : cai

实现 PCA 算法
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

def pca(X):
    # 归一化数据
    X = (X - X.mean()) / X.std()
    # 计算协方差矩阵
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # 实现 SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

# 将原始数据映射到低纬度空间
def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)

# 还原数据维度
def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)

dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex7data1.mat')
data = loadmat(dataPath)
X = data['X']

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.set_title('original data')
# plt.show()

U, S, V = pca(X)
# print('U=', U)
# print('S=', S)
# print('V=', V)

Z = project_data(X, U, 1)
# print('Z=', Z)
X_recoverd = recover_data(Z, U, 1)
print('recoveredX=', X_recoverd)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_recoverd[:, 0], X_recoverd[:, 1])
ax.set_title('recoverd data')
# plt.show()



