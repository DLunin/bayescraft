from scipy import inf
from scipy.stats import *
from scipy import stats, integrate
import scipy as sp
from scipy.special import gamma, gammaln
from scipy.stats._multivariate import _process_parameters, _squeeze_output
from ._multivariate import *
import numpy as np
from numpy.linalg import inv, cholesky, det
from numpy import exp, log
from sys import float_info

class multivariate_student_t_gen(rv_multivariate_continuous):
    r"""
    Mutivariate Student t-distribution random variable.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, mean, scale, shape):
        return multivariate_student_t_frozen(mean=mean, shape=shape, scale=scale)

    def _invgamma_pdf(self, x, a, scale):
        return (x**(-(a + 1))) * exp(-scale/x) / gamma(a)

    def _bad_pdf(self, x, mean, scale, shape, dim):
        part1 = (gamma(shape / 2. + dim / 2.) / gamma(shape / 2.))
        part2 = (1 / det(scale)) / ((shape*np.pi)**(dim / 2))
        part3_exp = 1 + ((1 / shape)*(x - mean).T*inv(scale)*(x - mean))[0,0]
        part3 = part3_exp ** (-(shape + dim)/2)
        #print(log(part1), log(part2), log(part3), sep=',')
        result = part1 * part2 * part3
        return part1 * part2 * part3

    def _pdf(self, x, mean, scale, shape, dim):
        eps = 1e-9

        part1 = (gamma(shape / 2. + dim / 2.) / gamma(shape / 2.))
        part1_ln = gammaln(shape / 2. + dim / 2.) - gammaln(shape / 2.)
        #print(abs(part1 - exp(part1_ln)))
        #assert abs(part1 - exp(part1_ln)) < eps

        part2 = (1 / det(scale)) / ((shape*np.pi)**(dim / 2))
        part2_ln = -log(det(scale)) - (log(shape*np.pi)*(dim / 2))
        #print(abs(part1 - exp(part1_ln)))
        #assert abs(part2 - exp(part2_ln)) < eps

        part3_exp = 1 + ((1 / shape)*(x - mean).T*inv(scale)*(x - mean))[0,0]
        part3 = part3_exp ** (-(shape + dim)/2)
        part3_ln = log(part3_exp) * (-(shape + dim)/2)
        #print(abs(part1 - exp(part1_ln)))
        #assert abs(part3 - exp(part3_ln)) < eps

        #print(part1, part2, part3, sep=',')
        result = exp(part1_ln + part2_ln + part3_ln)
        #(abs(self._bad_pdf(x, mean, scale, shape, dim) - result))
        #assert abs(self._bad_pdf(x, mean, scale, shape, dim) - result) < eps
        return result

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
        dim, mean, scale = _process_parameters(None, mean, scale)
        u = stats.chi2.rvs(df=shape, size=size)
        y = stats.multivariate_normal.rvs(mean=np.zeros(dim), cov=scale, size=size)
        return mean + np.array([y[i] * np.sqrt(shape / u)[i] for i in range(size)])
    
    def entropy(self, mean, scale, shape):
        raise NotImplementedError()

multivariate_student_t = multivariate_student_t_gen()

class multivariate_student_t_frozen(object):
    def __init__(self, mean, scale, shape):
        self.dim, self._mean, self.scale = _process_parameters(None, mean, scale)
        self.shape = shape
        self._student_t = multivariate_student_t_gen()

    def logpdf(self, x):
        return self._student_t.logpdf(x, mean=self._mean, scale=self.scale, shape=self.shape)

    def pdf(self, x):
        return self._student_t.pdf(x, mean=self._mean, scale=self.scale, shape=self.shape)

    def logcdf(self, x):
        return self._student_t.logcdf(x, mean=self._mean, scale=self.scale, shape=self.shape)

    def cdf(self, x):
        return self._student_t.cdf(x, mean=self._mean, scale=self.scale, shape=self.shape)

    def rvs(self, size=1):
        return self._student_t.rvs(mean=self._mean, scale=self.scale, shape=self.shape, size=size)

    def entropy(self):
        return self._student_t.entropy(mean=self._mean, scale=self.scale, shape=self.shape)

    @property
    def mean(self):
        if self.shape > 1:
            return self._mean
        return None

    @property
    def mode(self):
        return self.mean

    @property
    def covariance(self):
        return (self.shape / (self.shape - 2)) * self.scale
