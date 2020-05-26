# -*- coding: utf-8 -*- 

# Author Shuang Guo

import numpy as np

class Lasso:
    def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None


    def softThreshold(self, x, a):
        if (x < -a):
            return x + a
        elif (x > a):
            return x - a
        else:
            return 0.

    def fit(self, X, y):
        if (self.fit_intercept):
            X = np.column_stack((np.ones(len(X)),X))

        beta = np.zeros(X.shape[1])

        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:]))/(X.shape[0])

        for i in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                beta_tmp = beta.copy()
                beta_tmp[j] = 0.0
                r_j = y - np.dot(X, beta_tmp)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.alpha * X.shape[0]

                beta[j] = self.softThreshold(arg1, arg2) / (X[:, j]**2).sum()

                if self.fit_intercept:
                    beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:]))/(X.shape[0])

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta
                
        return self

    def predict(self, X):
        y = np.dot(X, self.coef_)
        if self.fit_intercept:
            y += self.intercept_*np.ones(len(y))
        return y
        