import numpy as np
import scipy as sp
from scipy import stats
from itertools import chain
from math import sqrt
from numpy.linalg import pinv

"""
This code provides united representation of probability distributions (or functions in general),
whether continuous or discrete, along with their domains. It allows to perform different
operations, like integration, in an unified way, without knowing the nature of underlying
distribution, so the code which uses only these operation would work for any such distribution.
"""

class DiscreteDomain:
    """
    A discrete domain -- a set of points.
    """
    def __init__(self, li=None):
        """
        :param li: list of point is the domain
        :return: None
        """
        self.values = set(li if li else [])

    def integrate(self, f):
        """
        Integrate over the domain.
        :param f: function to integrate
        :return: integral value
        """
        return sum(map(f, self.values))

    def __contains__(self, val):
        return val in self.values

class IntervalDomain:
    """
    An (open) interval on real line.
    """
    def __init__(self, begin=-np.inf, end=+np.inf):
        self.begin = begin
        self.end = end

    def integrate(self, f):
        """
        Integrate over the domain.
        :param f: function to integrate
        :return: integral value
        """
        return sp.integrate.quad(f, self.begin, self.end)[0]

    def __contains__(self, val):
        """
        Check if a point is in the domain.
        :param val: target point
        :return: True if point is in the domain, False otherwise
        """
        return val > self.begin and val < self.end

class UnionDomain:
    """
    Union of domains.
    """
    def __init__(self, *domains):
        flattened_domains = []
        for domain in domains:
            if isinstance(domain, UnionDomain):
                flattened_domains += domain.domains
            else:flattened_domains.append(domain)
        self.domains = flattened_domains

    def integrate(self, f):
        """
        Integrate over the domain.
        :param f: function to integrate
        :return: integral value
        """
        return sum(map(lambda domain: domain.integrate(f), self.domains))

    def __contains__(self, val):
        """
        Check if a point is in the domain.
        :param val: target point
        :return: True if point is in the domain, False otherwise
        """
        return any(map(lambda x: val in x, self.domains))

class ProductDomain:
    """
    Cartesian product of domains.
    """
    def __init__(self, *domains):
        flattened_domains = []
        for domain in domains:
            if isinstance(domain, ProductDomain):
                flattened_domains += domain.domains
            else:flattened_domains.append(domain)
        self.domains = flattened_domains

    def integrate(self, f):
        """
        Integrate over the domain.
        :param f: function to integrate
        :return: integral value
        """
        if len(self.domains) == 1:
            return self.domains[0].integrate(f)
        reduced_domain = ProductDomain(*self.domains[1:])
        return reduced_domain.integrate(lambda *args: self.domains[0].integrate(lambda x: f(x, *args)))

    def integrate_along(self, f, axis):
        """
        Integrate along one specified axis.
        :param f: function to integrate
        :param axis: selected axis
        :return: integral value
        """
        reduced_domain = ProductDomain(*(self.domains[:axis] + self.domains[axis+1:]))
        target_domain = self.domains[axis]
        g = lambda *args: target_domain.integrate(lambda x: f(*(args[:axis] + [x] + args[axis+1:])))
        return g, reduced_domain

    def __contains__(self, val):
        """
        Check if a point is in the domain.
        :param val: target point
        :return: True if point is in the domain, False otherwise
        """
        return all([val[i] in self.domains[i] for i in range(len(self.domains))])

    def __getitem__(self, pos):
        return self.domains[pos]

    def __iter__(self):
        return self.domains.__iter__()

class MathFunction:
    """
    Stores a Python function and its domain. Represents a mathematical function with
    domain information provided, so e.g. integration can be performed.
    """
    def __init__(self, f, domain):
        self.f = f
        self.domain = domain

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def integrate(self):
        """
        Integrate over all the domain.
        :return: integral value
        """
        return self.domain.integrate(self.f)

    def integrate_along(self, axis):
        """
        Integrate along one specified axis.
        :param axis: selected axis
        :return: integral value
        """
        return MathFunction(*self.domain.integrate_along(axis))

class MathDistribution:
    """
    Stores a distribution (scipy-style) and its domain.
    """

    def __init__(self, distr, domain):
        self.distr = distr
        self.domain = domain

    def __call__(self, *args, **kwargs):
        return self.distr(*args, **kwargs)

class MultivariateGaussianDistribution(MathDistribution):
    """
    Gaussian (normal) multivariate distribution.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = np.matrix(cov)
        self.dim = len(self.mean)
        assert self.cov.shape[0] == self.cov.shape[1]
        domain = ProductDomain([IntervalDomain(-np.inf, +np.inf) for i in range(len(self.mean))])
        super().__init__(stats.multivariate_normal(self.mean, self.cov), domain)

    def reduce(self, assignment):
        if all([x is None for x in assignment]):
            return MultivariateGaussianDistribution(self.mean, self.cov)
        # reordering variables, so that non-reduced variables go before reduced
        reduced_idx = [i for i in range(len(assignment)) if assignment[i] is not None]
        non_reduced_idx = [i for i in range(len(assignment)) if assignment[i] is None]
        x = np.matrix([assignment[idx] for idx in reduced_idx]).T
        new_idx = non_reduced_idx + reduced_idx
        mean1 = np.matrix([self.mean[idx] for idx in non_reduced_idx]).T
        mean2 = np.matrix([self.mean[idx] for idx in reduced_idx]).T
        cov11 = self.cov[non_reduced_idx][:, non_reduced_idx]
        cov22 = self.cov[reduced_idx][:, reduced_idx]
        cov12 = self.cov[non_reduced_idx][:, reduced_idx]
        mean = mean1 + cov12 * pinv(cov22) * (x - mean2)
        cov = cov11 - cov12 * pinv(cov22) * cov12.T
        return MultivariateGaussianDistribution(np.array(mean.T), cov)

    def marginalize(self, marginalized):
        non_marginalized = [i for i in range(self.dim)]
        mean = self.mean[non_marginalized]
        cov = self.cov[non_marginalized][:, non_marginalized]
        return MultivariateGaussianDistribution(mean, cov)

    def rvs(self, *args, **kwargs):
        return self.distr.rvs(*args, **kwargs)

class LinearGaussianDistribution:
    """
    Univariate gaussian distribution.
    """
    def __init__(self, w0, w, variance):
        self.w0 = w0
        self.w = w
        self.variance = variance
        self.dim = len(w) + 1

    @property
    def scale(self):
        return sqrt(self.variance)

    def pdf(self, x):
        x = np.atleast_1d(x)
        u = x[1:]
        x = x[0]
        return stats.norm.pdf(x, loc=np.dot(u, self.w) + self.w0, scale=self.scale)

    def rvs(self, size=1):
        assert self.dim == 1
        return stats.norm.rvs(size=size, loc=self.w0, scale=self.scale)

    def reduce(self, assignment):
        assert assignment[0] is None
        reduced = [i - 1 for i in range(1, len(assignment)) if assignment[i] is not None]
        non_reduced = [i - 1 for i in range(1, len(assignment)) if assignment[i] is None]
        reduced_values = np.array([x for x in assignment if x is not None])
        w0 = self.w0 + np.dot(self.w[reduced], reduced_values)
        w = self.w[non_reduced]
        return LinearGaussianDistribution(w0, w, self.variance)

    @staticmethod
    def mle(data):
        """
        Maximum Likelihood Estimation
        :param data: data in the form [(x, u0, u1, ... , un)], preferably numpy array
        :return: LinearGaussianDistribution with estimated parameters
        """
        data = np.asarray(data)
        u = data[:, 1:]
        x = data[:, 0]
        dim = data.shape[1]
        covs = np.matrix(np.atleast_2d(np.cov(np.transpose(data))), copy=False)
        means = np.mean(data, axis=0)
        A = np.matrix(np.zeros((dim, dim)))
        A[0, 0] = 1
        for i in range(1, dim):
            for j in range(1, i + 1):
                A[i, j] = covs[i, j]
                A[j, i] = covs[i, j]
        b = np.zeros(dim-1)
        for i in range(1, dim):
            b[i-1] = covs[0, i]
        if data.shape[1] > 1:
            w = np.linalg.solve(A[1:, 1:], b)
        else:
            w = np.array([])
        w0 = means[0] - np.dot(means[1:], w)
        mw = np.matrix(w).T
        if data.shape[1] > 1:
            variance = covs[0, 0] + (mw.T * covs[1:, 1:] * mw).sum(axis=1)
        else:
            variance = covs[0, 0]
        return LinearGaussianDistribution(w0, w, variance)