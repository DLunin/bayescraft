import networkx as nx
import numpy as np
from numpy import log, exp
import scipy as sp
import pymc
from scipy import stats
import emcee
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mutual_info_score
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from itertools import *
from NPEET import mi
from math import sqrt
import math

from .utility import *

def monte_carlo_integration(distr_rvs, f, initial_chunk_size=100, chunk_size=10, eps=0.1):
    current_size = initial_chunk_size
    prev = 0.
    current = sum(map(f, distr_rvs(size=current_size))) / current_size
    i = 0
    while abs(current - prev) > eps:
        i += 1
        prev = current
        current = current * current_size / (current_size + chunk_size)
        current_size += chunk_size
        current += sum(map(f, distr_rvs(size=chunk_size))) / current_size
    return current

class KDE:
    def __init__(self, data, h=1):
        try:
            self.dim = len(data[0])
        except TypeError:
            self.dim = 1
        except IndexError:
            raise Exception("data is empty")
        self.h = h
        self.distrs = [stats.multivariate_normal(mean=x, cov=np.eye(self.dim) * h) for x in data]

    @property
    def n(self):
        return len(self.distrs)

    def __call__(self, x):
        return sum(map(lambda distr: distr.pdf(x), self.distrs)) / self.n / self.h


mutinfo_density_estimator = lambda data: gaussian_kde(data)
def mutual_information_naive(data1, data2, debug=False):
    """"""
    """
    def is_discrete(col):
        for x in col:
            if abs(int(x) - x) > 1e-9:
                return False
        return True

    def infer_data_domain(data):
        def infer_datacol_domain(col):
            if not is_discrete(col):
                return IntervalDomain(-np.inf, +np.inf)
            return DiscreteDomain(col)
        return ProductDomain(*list(map(infer_datacol_domain(data[:, i]), range(data.shape[1]))))

    def density(data):
        discrete = lmap(is_discrete, [data[:, i] for i in range(data.shape[1])])
        domain = infer_
    """


    data1 = np.transpose(np.array(data1))
    data2 = np.transpose(np.array(data2))
    distr1 = mutinfo_density_estimator(data1)
    distr2 = mutinfo_density_estimator(data2)
    distr12 = mutinfo_density_estimator(np.vstack([data1, data2]))
    if debug:
        plot_distr(distr1)
        plot_distr(distr2)
        plot_distr(distr12)
    dim1 = data1.shape[0]
    dim2 = data2.shape[0]

    def f(x):
        return np.log(distr12(x)) - np.log(distr1(x[:dim1])) - np.log(distr2(x[dim1:]))

    return monte_carlo_integration(lambda size=1: np.transpose(distr12.resample(size=size)), f)

mutual_information = lambda *args, debug=None, **kwargs: mi(*args, **kwargs)

def pairwise_mutual_info(data, v1, v2):
    return mutual_information(data[:, v1:v1+1], data[:, v2:v2+1])