import numpy as np
from numpy.linalg import inv, pinv
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

class LinearRegressionKnownVariance:
    def __init__(self, dim=None, w_0=None, V_0=None, variance=None):
        self.dim = dim
        self.variance = variance
        self.w_0 = to_column_vector(w_0)
        self.V_0 = to_matrix(V_0)

        self.w_N = self.w_0
        self.V_N = self.V_0

    def fit(self, X, y):
        var = self.variance
        V_0 = self.V_0

        V_N = var*inv(var*inv(V_0) + X.T*X)
        w_N = V_N*inv(V_0)*self.w_0 + (1 / var)*V_N*X.T*y

        self.V_N = V_N
        self.w_N = w_N

    @property
    def w(self):
        return stats.multivariate_normal(mean=np.array(self.w_N.T, copy=False)[0], cov=self.V_N)

    def posterior_predictive(self, X):
        X = np.matrix(X)
        return stats.multivariate_normal(mean=X*self.w_N, cov=self.variance*matrix_eye(self.dim) + X*self.V_N*X.T)

    def predict(self, X):
        return np.array(np.matrix(X)*self.w_N)

class LinearRegressionUnknownVariance:
    @staticmethod
    def g_prior(X, g=1e+50):
        X = to_matrix(X)
        dim = X.shape[1]
        a_0 = b_0 = 0
        w_0 = np.zeros((dim, 1))
        V_0 = g*(inv(X.T*X))
        return LinearRegressionUnknownVariance(dim=dim, w_0=w_0, V_0=V_0, a_0=a_0, b_0=b_0)

    @staticmethod
    def uninformative_prior(dim):
        g = 1e+6
        a_0 = b_0 = 0
        w_0 = np.zeros((dim, 1))
        V_0 = g*np.matrix(np.ones((dim, dim)), copy=False)
        return LinearRegressionUnknownVariance(dim=dim, w_0=w_0, V_0=V_0, a_0=a_0, b_0=b_0)

    def __init__(self, dim=None, w_0=None, V_0=None, a_0=None, b_0=None):
        self.dim = dim
        self.w_0 = to_column_vector(w_0)
        self.V_0 = to_matrix(V_0)
        self._inv_V_0 = pinv(self.V_0)
        self.a_0 = a_0
        self.b_0 = b_0
        self.reset()

    def reset(self):
        self.w_N = self.w_0
        self.V_N = self.V_0
        self.a_N = self.a_0
        self.b_N = self.b_0
        self._inv_V_N = self._inv_V_0
        self._b_init = self.w_0.T*self._inv_V_0*self.w_0

    @property
    def w(self):
        return multivariate_student_t(mean=self.w_N, scale=(self.b_N / self.a_N)*self.V_N, shape=2*self.a_N)

    @property
    def var(self):
        return stats.invgamma(a=self.a_N, scale=self.b_N)

    @property
    def w_var(self):
        return normal_inverse_gamma(w_0=self.w_N, V_0=self.V_N, a_0=self.a_N, b_0=self.b_N)

    def fit(self, X, y):
        self.reset()
        X = to_matrix(X)
        y = to_column_vector(y)

        n = X.shape[0]
        V_0 = self.V_0
        w_0 = self.w_0
        inv_V_N = self._inv_V_0 + X.T*X
        V_N = inv(inv_V_N)
        w_N = V_N*(self._inv_V_0*w_0 + X.T*y)
        a_N = self.a_0 + n/2
        b_init = self._b_init + y.T*y
        b_N = to_scalar(self.b_0 + 0.5*(b_init - w_N.T*inv(V_N)*w_N))

        self.w_N = w_N
        self.b_N = b_N
        self.a_N = a_N
        self.V_N = V_N
        self._inv_V_N = inv_V_N
        self._b_init = b_init

    def update(self, x, y):
        x = to_column_vector(x)
        y = to_scalar(y)

        w_old = self.w_N
        w_0 = self.w_0
        V_old = self.V_N
        V_0 = self.V_0
        b_0 = self.b_0

        K = V_old * x * x.T
        g = np.trace(K)
        M = matrix_eye(K.shape[0]) - (1 / (1 + g))*K
        V_N = M*V_old
        inv_V_N = self._inv_V_N + x*x.T
        w_N = M*(w_old + V_old*x*y)
        b_init = self._b_init + y*y
        b_N = to_scalar(b_0 + 0.5*(b_init - w_N.T*inv_V_N*w_N))
        a_N = self.a_N + 0.5

        self.a_N = a_N
        self.b_N = b_N
        self.w_N = w_N
        self.V_N = V_N
        self._inv_V_N = inv_V_N
        self._b_init = b_init

    def posterior_predictive(self, X):
        X = np.matrix(X)
        return multivariate_student_t(mean=X*self.w_N,
                                      scale=(self.b_N / self.a_N)*(np.eye(X.shape[0]) + X*self.V_N*X.T),
                                      shape=2*self.a_N)

    def predict(self, X):
        X = np.matrix(X)
        return to_1d_array(self.posterior_predictive(X).mode)

LinearRegression = LinearRegressionUnknownVariance

