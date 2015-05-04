import numpy as np
from numpy.linalg import inv
from numpy import log, sqrt
import scipy as sp
import matplotlib.pyplot as plt
import sys
import sklearn.linear_model as linear
import scipy.stats as stats
from .stats import *
from collections import ChainMap
import scipy

def prettify_data(X, y=None):
    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    X = np.matrix(X, copy=False)
    if y is not None:
        y = np.matrix(y)
        if y.shape[0] == 1 and y.shape[1] != 1:
            y = y.T
        assert X.shape[0] == y.shape[0]
        return X, y
    else:
        return X
        
class LinearRegression:
    def __init__(self, variance=None, tau=sys.float_info.max, w_0=None, V_0=None, a_0=None, b_0=None):
        self.variance = variance
        self.tau = tau
        self.w_0 = w_0
        self.V_0 = V_0
        self.a_0 = a_0
        self.b_0 = b_0
    
    def fit(self, X, y):
        X, y = prettify_data(X, y)
        xdim = X.shape[1]
        n = X.shape[0]
        if self.V_0 is None:
            self.V_0 = np.matrix(np.eye(xdim), copy=False)*self.tau
        if self.w_0 is None:
            self.w_0 = np.matrix(np.zeros((xdim, 1)), copy=False)
        if self.variance is None:
            return self._fit_with_unknown_variance(X, y)
        else:
            return self._fit_with_known_variance(X, y, self.variance, w_0=self.w_0, V_0=self.V_0)

    def predict(self, X):
        X = prettify_data(X)
        if type(self.w) is stats.multivariate_normal:
            return np.array([self._predict_normal(x) for x in X])
        else: # unknown variance scenario
            return self.posterior_predictive(X)
        
    def _fit_with_known_variance(self, X, y, var, w_0, V_0):
        xdim = X.shape[1]
        n = X.shape[0]

        V_N = var*inv(var*inv(V_0) + X.T*X)
        w_N = V_N*inv(V_0)*w_0 + (1 / var)*V_N*X.T*y
        
        self.w = stats.multivariate_normal(mean=np.array(w_N.T, copy=False)[0], cov=V_N)
    
    def _fit_with_unknown_variance(self, X, y):
        n = X.shape[0]
        V_0 = self.V_0
        w_0 = self.w_0
        V_N = inv(inv(V_0) + X.T*X)
        w_N = V_N*(inv(V_0)*w_0 + X.T*y)
        a_N = self.a_0 + n/2
        b_N = self.b_0 + 0.5*(w_0.T*inv(V_0)*w_0 + y.T*y - w_N.T*inv(V_N)*w_N)
        self.w_var = normal_inverse_gamma(w_0=w_N, V_0=V_N, a_0=a_N, b_0=b_N)
        self.w_N = w_N
        self.b_N = b_N
        self.a_N = a_N
        self.V_N = V_N
        self.var = invgamma(a=a_N, scale=b_N)
        self.w = multivariate_student_t(mean=w_N, scale=(b_N / a_N)*V_N, shape=2*a_N)
        self.posterior_predictive = lambda X: multivariate_student_t(mean=X*w_N, scale=(b_N / a_N)*(np.eye(X.shape[0]) + X*V_N*X.T))
        
    def _predict_normal(self, x):
        w_N = self.w.mean()
        V_N = self.w.var()
        variance_N = self.variance
        return stats.multivariate_normal(mean=w_N.T*x, cov=variance_N)
