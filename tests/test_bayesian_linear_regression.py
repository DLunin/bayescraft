import bayescraft as b
import numpy as np
import scipy as sp
from scipy.linalg import inv, pinv

def test_confidence_interval_with_unknown_variance():
    n = 1000

    def generate_data(n):
        X, y = [], []
        for i in range(n):
            X.append([np.random.random()*10])
            y.append(np.random.normal(5*X[-1][0], np.sqrt(10)))
        return np.array(X), np.array(y)


    counter = 0
    for i in range(n):
        X, y = generate_data(n=1000)
        X, y = b.prettify_data(X, y)

        """ using g-prior with g-> +oo """
        a_0 = b_0 = 0 
        g = 1e+50
        w_0 = np.zeros((1, 1))
        V_0 = g*(inv(X.T*X))
        print(V_0)
        lr = b.LinearRegression(w_0=w_0, a_0=a_0, b_0=b_0, V_0=V_0)

        lr.fit(X, y)
        coef = lr.w.mode[0]
        cov = lr.w.covariance[0,0]
        if (coef - 1.96*np.sqrt(cov)) < 5 < (coef + 1.96*np.sqrt(cov)):
            counter += 1
    fraction = counter / n
    print(fraction)
    assert abs(fraction - 0.95) < 0.01
