import numpy as np
from numpy.linalg import inv
from numpy import log, sqrt
import scipy as sp
import matplotlib.pyplot as plt
import sys
import sklearn.linear_model as linear
import scipy.stats as stats
from .stats import *
from .utils.data_formats import *
import scipy

def matrix_eye(n):
    return np.matrix(np.eye(n), copy=False)

# class LinearRegressionKnownVariance:
#     def __init__(self, w_0=None, V_0=None, variance=None, tau=sys.float_info.max):
#         self.variance = variance
#         self.tau = tau
#         self.w_0 = to_column_vector(w_0)
#         self.V_0 = to_matrix(V_0)
#
#     def fit(self, X, y):
#         var = self.variance
#         V_0 = self.V_0
#
#
#         V_N = var*inv(var*inv(V_0) + X.T*X)
#         w_N = V_N*inv(V_0)*w_0 + (1 / var)*V_N*X.T*y
#
#         self.w = stats.multivariate_normal(mean=np.array(w_N.T, copy=False)[0], cov=V_N)
        
class LinearRegression:
    def __init__(self, variance=None, tau=sys.float_info.max, w_0=None, V_0=None, a_0=None, b_0=None):
        self.variance = variance
        self.tau = tau
        self.w_0 = to_column_vector(w_0)
        self.V_0 = to_matrix(V_0)
        self.a_0 = a_0
        self.b_0 = b_0
        self.fitted = False
    
    def fit(self, X, y):
        X = to_matrix(X)
        y = to_column_vector(y)

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

    def update(self, x, y):
        if self.variance is None:
            if self.fitted:
                x = to_column_vector(x)
                y = to_scalar(y)
                return self._update_with_unknown_variance(x, y)
            else:
                return self.fit([x], y)
        else:
            raise NotImplementedError()

    def predict(self, X):
        X = to_matrix(X)
        if type(self.w) is stats.multivariate_normal:
            return np.array([self._predict_normal(x) for x in X])
        else: # unknown variance scenario
            return self.posterior_predictive(X)

        
    def _fit_with_known_variance(self, X, y, var, w_0, V_0):
        self.fitted = True
        xdim = X.shape[1]
        n = X.shape[0]

        V_N = var*inv(var*inv(V_0) + X.T*X)
        w_N = V_N*inv(V_0)*w_0 + (1 / var)*V_N*X.T*y
        
        self.w = stats.multivariate_normal(mean=np.array(w_N.T, copy=False)[0], cov=V_N)
    
    def _fit_with_unknown_variance(self, X, y):
        self.fitted = True
        n = X.shape[0]
        V_0 = self.V_0
        w_0 = self.w_0
        inv_V_N = inv(V_0) + X.T*X
        V_N = inv(inv_V_N)
        w_N = V_N*(inv(V_0)*w_0 + X.T*y)
        a_N = self.a_0 + n/2
        b_init = w_0.T*inv(V_0)*w_0 + y.T*y
        b_N = to_scalar(self.b_0 + 0.5*(b_init - w_N.T*inv(V_N)*w_N))
        self.w_var = normal_inverse_gamma(w_0=w_N, V_0=V_N, a_0=a_N, b_0=b_N)
        self.w_N = w_N
        self.b_N = b_N
        self.a_N = a_N
        self.V_N = V_N
        self.inv_V_N = inv_V_N
        self.b_init = b_init
        self.var = stats.invgamma(a=a_N, scale=b_N)
        self.w = multivariate_student_t(mean=w_N, scale=(b_N / a_N)*V_N, shape=2*a_N)
        mean=X*w_N
        scale=(b_N / a_N)*(np.eye(X.shape[0]) + X*V_N*X.T)
        shape=2*a_N
        self.posterior_predictive = lambda X: multivariate_student_t(mean=X*w_N, scale=(b_N / a_N)*(np.eye(X.shape[0]) + X*V_N*X.T), shape=2*a_N)

    def _update_with_unknown_variance(self, x, y):
        assert self.fitted

        w_old = self.w_N
        w_0 = self.w_0
        V_old = self.V_N
        V_0 = self.V_0
        b_0 = self.b_0

        K = V_old * x * x.T
        g = np.trace(K)
        M = matrix_eye(K.shape[0]) - (1 / (1 + g))*K
        V_N = M*V_old
        inv_V_N = self.inv_V_N + x*x.T
        w_N = M*(w_old + V_old*x*y)
        b_init = self.b_init + y*y
        b_N = to_scalar(b_0 + 0.5*(b_init - w_N.T*inv_V_N*w_N))
        a_N = self.a_N + 0.5

        self.a_N = a_N
        self.b_N = b_N
        self.w_N = w_N
        self.V_N = V_N
        self.inv_V_N = inv_V_N
        self.b_init = b_init

        self.var = stats.invgamma(a=a_N, scale=b_N)
        self.w = multivariate_student_t(mean=w_N, scale=(b_N / a_N)*V_N, shape=2*a_N)
        self.posterior_predictive = lambda X: multivariate_student_t(mean=X*w_N, scale=(b_N / a_N)*(np.eye(X.shape[0]) + X*V_N*X.T), shape=2*a_N)



    def _predict_normal(self, x):
        w_N = self.w.mean()
        V_N = self.w.var()
        variance_N = self.variance
        return stats.multivariate_normal(mean=w_N.T*x, cov=variance_N)

def g_prior(X, g=1e+50):
    X = to_matrix(X)
    dim = X.shape[1]
    a_0 = b_0 = 0
    w_0 = np.zeros((dim, 1))
    V_0 = g*(inv(X.T*X))
    return LinearRegression(w_0=w_0, V_0=V_0, a_0=a_0, b_0=b_0)