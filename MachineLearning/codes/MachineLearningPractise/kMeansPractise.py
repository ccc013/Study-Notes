#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/21 16:35
@Author  : cai

实现 K-Means 聚类算法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# 寻址最近的中心点
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            # 计算每个训练样本和中心点的距离
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                # 记录当前最短距离和其中心的索引值
                min_dist = dist
                idx[i] = j

    return idx

# 计算聚类中心
def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        # 计算下一个聚类中心，这里简单的将该类中心的所有数值求平均值作为新的类中心
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids

# 初始化聚类中心
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    # 随机初始化 k 个 [0,m]的整数
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids

# 实现 kmeans 算法
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    # 聚类中心的数目
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids

dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex7data2.mat')
data = loadmat(dataPath)
X = data['X']

initial_centroids = init_centroids(X, 3)
# print(initial_centroids)
# idx = find_closest_centroids(X, initial_centroids)
# print(idx)

# print(compute_centroids(X, idx, 3))

idx, centroids = run_k_means(X, initial_centroids, 10)
# 可视化聚类结果
cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
# ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
# ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
# ax.legend()
# plt.show()

# 载入一张测试图片，进行测试
imageDataPath = os.path.join('E:\\ipython-notebooks\\data', 'bird_small.mat')
image = loadmat(imageDataPath)
# print(image)

A = image['A']
print(A.shape)

# 对图片进行归一化
A = A / 255.

# 重新调整数组的尺寸
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
# 随机初始化聚类中心
initial_centroids = init_centroids(X, 16)
# 运行聚类算法
idx, centroids = run_k_means(X, initial_centroids, 10)

# 得到最后一次的最近中心点
idx = find_closest_centroids(X, centroids)
# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int), :]
# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

plt.imshow(X_recovered)





