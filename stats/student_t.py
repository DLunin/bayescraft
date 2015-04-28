from scipy import inf
from scipy.stats import *
from scipy import stats, integrate
import scipy as sp
from scipy.special import gamma
from scipy.stats._multivariate import _process_parameters, _squeeze_output
from ._multivariate import *
import numpy as np
from numpy.linalg import inv, cholesky, det
from numpy import exp
from sys import float_info

class multivariate_student_t_gen(rv_multivariate_continuous):
    r"""
    Mutivariate Student t-distribution random variable.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, mean, shape, scale):
        return multivariate_student_t_frozen(mean=mean, shape=shape, scale=scale)

    def _invgamma_pdf(self, x, a, scale):
        return (x**(-(a + 1))) * exp(-scale/x) / gamma(a)

    def _pdf(self, x, mean, scale, shape, dim):
        part1 = (gamma(shape / 2. + dim / 2.) / gamma(shape / 2.))
        part2 = (1 / det(scale)) / ((shape*np.pi)**(dim / 2))
        part3_exp = 1 + ((1 / shape)*(x - mean).T*inv(scale)*(x - mean))[0,0]
        part3 = part3_exp ** (-(shape + dim)/2)
        return part1 * part2 * part3

    def pdf(self, x, mean, scale, shape):
        dim, mean, scale = _process_parameters(None, mean, scale)
        x = np.matrix(x, copy=False).T
        mean = np.matrix(mean, copy=False).T
        return self._pdf(x, mean=mean, scale=scale, shape=shape, dim=dim)

    def cdf(self, x, mean, scale, shape):
        dim, mean, scale = _process_parameters(None, mean, scale)
        x = np.matrix(x, copy=False).T
        mean = np.matrix(mean, copy=False).T
        def temp(*args):
            result = self._pdf(np.matrix(args).T, mean=mean, scale=scale, shape=shape, dim=dim)
            return result

        return sp.integrate.nquad(lambda *args: self._pdf(np.array(args), mean, scale, shape, dim),
        # return sp.integrate.nquad(temp,
                                  [(-sp.inf, x_i) for x_i in x])[0]

    def rvs(self, mean, scale, shape, size=1):
        raise NotImplementedError()
        dim, mean, scale = _process_parameters(None, mean, scale)
        result = None
        return _squeeze_output(result)

    def entropy(self, mean, scale, shape):
        raise NotImplementedError()

multivariate_student_t = multivariate_student_t_gen()

class multivariate_student_t_frozen(object):
    def __init__(self, mean, scale, shape):
        self.dim, self.mean, self.scale = _process_parameters(None, mean, scale)
        self.shape = shape
        self._student_t = multivariate_student_t_gen()

    def logpdf(self, x):
        return self._student_t.logpdf(x, mean=self.mean, scale=self.scale, shape=self.shape)

    def pdf(self, x):
        return self._student_t.pdf(x, mean=self.mean, scale=self.scale, shape=self.shape)

    def logcdf(self, x):
        return self._student_t.logcdf(x, mean=self.mean, scale=self.scale, shape=self.shape)

    def cdf(self, x):
        return self._student_t.cdf(x, mean=self.mean, scale=self.scale, shape=self.shape)

    def rvs(self, size=1):
        return self._student_t.rvs(mean=self.mean, scale=self.scale, shape=self.shape, size=size)

    def entropy(self):
        return self._student_t.entropy(mean=self.mean, scale=self.scale, shape=self.shape)
