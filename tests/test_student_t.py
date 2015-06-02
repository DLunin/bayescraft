import bayescraft.stats as bstats
import scipy.stats as stats
import numpy as np
from numpy.linalg import inv
import scipy as sp

def test_approaches_normal():
    eps = 1e-6
    df = 1e+9
    dim = 5
    n_points = 1000
    t_distr = bstats.multivariate_student_t(mean=np.zeros(dim), scale=np.eye(dim), shape=df)
    normal_distr = stats.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    for i in range(n_points):
        point = np.random.rand(dim)
        print(t_distr.pdf(point))
        print(normal_distr.pdf(point))
        assert abs(t_distr.pdf(point) - normal_distr.pdf(point)) < eps

