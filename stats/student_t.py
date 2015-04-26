from scipy import inf
from scipy.stats import *
from scipy import stats, integrate
import scipy as sp
from scipy.special import gamma
from scipy.stats._multivariate import _process_parameters, _squeeze_output
from ._multivariate import *
import numpy as np
from numpy import exp
from sys import float_info

class multivariate_student_t_gen(rv_multivariate_continuous):
    r"""
    A Normal Inverse Gamma (NIG) random variable.
    """

    def __init__(self):
        self._mnorm = multivariate_normal
        self._invgamma = invgamma

    def __call__(self, w_0=None, V_0=None, a_0=0, b_0=0):
        return multivariate_student_t_frozen(w_0=w_0, V_0=V_0, a_0=a_0, b_0=b_0)

    def _invgamma_pdf(self, x, a, scale):
        return (x**(-(a + 1))) * exp(-scale/x) / gamma(a)

    def _pdf(self, w, var, w_0, V_0, a_0, b_0):
        return self._mnorm.pdf(w, mean=w_0, cov=var*V_0)*self._invgamma.pdf(var, a=a_0, scale=b_0)

    def pdf(self, x, w_0, V_0, a_0, b_0):
        dim, w_0, V_0 = _process_parameters(None, w_0, V_0)
        return self._pdf(x[:-1], x[-1], w_0, V_0, a_0, b_0)

    def logpdf(self, x, w_0, V_0, a_0, b_0):
        var = x[-1]
        w = x[:-1]
        return self._mnorm.logpdf(w, mean=w_0, cov=var*V_0) + self._invgamma.logpdf(var, a=a_0, scale=b_0)

    def cdf(self, x, w_0, V_0, a_0, b_0):
        dim, w_0, V_0 = _process_parameters(None, w_0, V_0)
        var = x[-1]
        w = x[:-1]
        return sp.integrate.nquad(lambda *args: self._pdf(np.array(args[:-1]), args[-1], w_0, V_0, a_0, b_0),
                                  [(-sp.inf, x) for x in w] + [(0, var)])[0]

    def rvs(self, w_0, V_0, a_0, b_0, size=1):
        dim, w_0, V_0 = _process_parameters(None, w_0, V_0)
        ig = self._invgamma.rvs(a=a_0, scale=b_0, size=size)
        result = np.vstack([np.append(self._mnorm.rvs(mean=w_0, cov=var*V_0, size=1), var) for var in ig])
        return _squeeze_output(result)

    def entropy(self, w_0, V_0, a_0, b_0):
        raise NotImplementedError()

multivariate_student_t = multivariate_student_t_gen()

def frozen(distribution):
    class frozen_rv(object):
        def __init__(self, **params):
            self.__dict__.update(params)
            self._param_list = list(params.keys())

        def logpdf(self, x):
            return self._nig.logpdf(x, **{p: getattr(self, p) for p in self._param_list})

        def pdf(self, x):
            return self._nig.pdf(x, **{p: getattr(self, p) for p in self._param_list})

        def logcdf(self, x):
            return self._nig.logcdf(x, **{p: getattr(self, p) for p in self._param_list})

        def cdf(self, x):
            return self._nig.cdf(x, **{p: getattr(self, p) for p in self._param_list})

        def rvs(self, size=1):
            return self._nig.rvs(size=size, **{p: getattr(self, p) for p in self._param_list})

        def entropy(self):
            return self._nig.entropy(**{p: getattr(self, p) for p in self._param_list})
    return frozen_rv