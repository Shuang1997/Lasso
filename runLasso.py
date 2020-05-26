# 
# Copyright (c) 2016-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import numpy as np
#from pandas import Series,DataFrame

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import r2_score
import sklearn.datasets as datasets

import matplotlib.pyplot as plt

from Lasso import Lasso

boston = datasets.load_boston()
#iris = datasets.load_iris()
#diabetes = datasets.load_diabetes()
#digits = datasets.load_digits()
#linnerud = datasets.load_linnerud()
#wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()

#data = boston.data
#target = boston.target

data = boston.data
target = boston.target

#训练数据
X_train = data[:480]
Y_train = target[:480]
#测试数据
x_test = data[480:]
y_true = target[480:]

'''
df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() # 基準化
X = df.iloc[:, :13].values
'''

line = LinearRegression()
ridge = Ridge()
lasso = Lasso(alpha=0.01, max_iter=1000)

line.fit(X_train,Y_train)
ridge.fit(X_train,Y_train)
lasso.fit(X_train,Y_train)

line_y_pre=line.predict(x_test)
ridge_y_pre=ridge.predict(x_test)
lasso_y_pre=lasso.predict(x_test)

line_score=r2_score(y_true,line_y_pre)
ridge_score=r2_score(y_true,ridge_y_pre)
lasso_score=r2_score(y_true,lasso_y_pre)
print(line_score,ridge_score,lasso_score)

plt.plot(y_true,label='True')
plt.plot(line_y_pre,label='Line')
plt.plot(ridge_y_pre,label='Ridge')
plt.plot(lasso_y_pre,label='Lasso')
plt.legend()

plt.show()

#model.fit(X, y)

#print(model.intercept_)
#print(model.coef_)

