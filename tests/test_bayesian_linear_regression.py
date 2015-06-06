import bayescraft as b
import numpy as np
from numpy import sqrt
import scipy as sp
from scipy import stats
from scipy.linalg import inv, pinv
import matplotlib.pyplot as plt

def generate_data(n, dim=1, x_range=(0, 10), coef_range=(1, 10), noise_var=1):
    global tick
    X, y = [], []

    coef_mid = (coef_range[0] + coef_range[1]) / 2.
    coef_range = coef_range[1] - coef_range[0]
    x_mid = (x_range[0] + x_range[1]) / 2.
    x_range = x_range[1] - x_range[0]

    coef = (np.random.rand(dim) - 0.5)*coef_range + coef_mid
    for i in range(n):
        X.append((np.random.rand(dim) - 0.5)*x_range + x_mid)
        y.append(np.dot(X[-1], coef) + np.random.randn(dim)*sqrt(noise_var))
    X = np.array(X)
    y = np.array(y)
    return X, y, coef

def test_confidence_interval_with_unknown_variance():
    n = 1000
    counter = 0
    for i in range(n):
        X, y, c = generate_data(n=100)
        X = np.matrix(X)
        y = np.matrix(y)

        """ using g-prior with g-> +oo """
        a_0 = b_0 = 0 
        g = 1e+50
        w_0 = np.zeros((1, 1))
        V_0 = g*(inv(X.T*X))
        lr = b.LinearRegressionUnknownVariance(dim=1, w_0=w_0, a_0=a_0, b_0=b_0, V_0=V_0)

        lr.fit(X, y)
        coef = lr.w.mode[0]
        cov = lr.w.covariance[0,0]
        if (coef - 1.96*np.sqrt(cov)) < c[0] < (coef + 1.96*np.sqrt(cov)):
            counter += 1
    fraction = counter / n
    print(fraction)
    assert abs(fraction - 0.95) < 0.02
