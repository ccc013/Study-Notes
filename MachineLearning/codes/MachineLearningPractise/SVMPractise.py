#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/21 15:52
@Author  : cai

实现支持向量机算法
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import os

# 定义一个高斯核函数
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))


dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex6data1.mat')
raw_data = loadmat(dataPath)
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
# 可视化数据
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.legend()
# plt.show()

# 使用C=1
# svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
# print(svc)
# svc.fit(data[['X1', 'X2']], data['y'])
# print('C=1, score = ', svc.score(data[['X1', 'X2']], data['y']))

# data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
# 可视化SVM的决策边界
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
# ax.set_title('SVM (C=1) Decision Confidence')
# plt.show()

# 使用C=100
# svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
# print(svc)
# svc2.fit(data[['X1', 'X2']], data['y'])
# print('C=100, score = ', svc2.score(data[['X1', 'X2']], data['y']))
# data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
# ax.set_title('SVM (C=100) Decision Confidence')
# plt.show()

# 测试高斯核函数
# x1 = np.array([1.0, 2.0, 1.0])
# x2 = np.array([0.0, 4.0, -1.0])
# sigma = 2
# print(gaussian_kernel(x1, x2, sigma))

# 第二个数据库
# dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex6data2.mat')
# raw_data = loadmat(dataPath)
# data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
# data['y'] = raw_data['y']
#
# positive = data[data['y'].isin([1])]
# negative = data[data['y'].isin([0])]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
# ax.legend()
# plt.show()

# 使用RBF核函数
# svc = svm.SVC(C=100, gamma=10, probability=True)
# svc.fit(data[['X1', 'X2']], data['y'])
# data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:, 0]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
# plt.show()

# 第三个数据库，给出了训练集和验证集
# dataPath = os.path.join('E:\\ipython-notebooks\\data', 'ex6data3.mat')
# raw_data = loadmat(dataPath)
# # print(raw_data)
# X = raw_data['X']
# Xval = raw_data['Xval']
# y = raw_data['y'].ravel()
# yval = raw_data['yval'].ravel()
#
# C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
# gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
#
# best_score = 0
# best_params = {'C': None, 'gamma': None}
# 多次实验，找到最佳的参数C和gamma值
# for C in C_values:
#     for gamma in gamma_values:
#         svc = svm.SVC(C=C, gamma=gamma)
#         svc.fit(X, y)
#         score = svc.score(Xval, yval)
#
#         if score > best_score:
#             best_score = score
#             best_params['C'] = C
#             best_params['gamma'] = gamma

# print('best_score = {0}, best_params={1}'.format(best_score,best_params))

# 最后是给出垃圾邮件的数据库,使用SVM对垃圾邮件进行筛选
dataPath1 = os.path.join('E:\\ipython-notebooks\\data', 'spamTrain.mat')
dataPath2 = os.path.join('E:\\ipython-notebooks\\data', 'spamTest.mat')

spam_train = loadmat(dataPath1)
spam_test = loadmat(dataPath2)

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()
# 输出数据的维度，每个邮件都是由1899个单词组成，训练集有4000个样本，测试集有1000个样本
# print(X.shape, y.shape, Xtest.shape, ytest.shape)

spamSvc = svm.SVC()
spamSvc.fit(X, y)
print('Test accuracy = {0}%'.format(np.round(spamSvc.score(Xtest, ytest) * 100), 2))
