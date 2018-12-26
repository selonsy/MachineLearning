# -*- coding: utf-8 -*-
"""
决策边界
"""

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import scipy.io as sio 
from sklearn.linear_model import LogisticRegressionCV

#matlab文件名 
f_w1='w1.mat'
f_w2='w2.mat'
f_R1='R1.mat'
f_R2='R2.mat'

w1=sio.loadmat(f_w1)
w2=sio.loadmat(f_w2)
R1=sio.loadmat(f_R1)
R2=sio.loadmat(f_R2)

_X=sio.loadmat('w.mat')
_R=sio.loadmat('R.mat')
X=_X['w']
R=_R['R'][0]

plt.plot(w1['w1'][:,0],w1['w1'][:,1],'ro',w2['w2'][:,0],w2['w2'][:,1],'g*')
# plt.show()

# 画决策边界
def plot_decision_boundary(pred_func):
 
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 
    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
 
    # 然后画出图
    plt.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=R, cmap=plt.cm.Spectral)

clf = LogisticRegressionCV()
clf.fit(X, R)

plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()