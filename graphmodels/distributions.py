import numpy as np
import scipy as sp
from scipy import stats
from itertools import chain
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

class GaussianDistribution(MathDistribution):
    """
    Gaussian (normal) multivariate distribution.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = np.matrix(cov)
        assert self.cov.shape[0] == self.cov.shape[1]
        domain = ProductDomain([IntervalDomain(-np.inf, +np.inf) for i in range(len(self.mean))])
        super().__init__(stats.multivariate_normal(self.mean, self.cov), domain)

    def reduce(self, assignment):
        if all([x is None for x in assignment]):
            return GaussianDistribution(self.mean, self.cov)
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
        return GaussianDistribution(np.array(mean.T), cov)

    def rvs(self, *args, **kwargs):
        return self.distr.rvs(*args, **kwargs)
