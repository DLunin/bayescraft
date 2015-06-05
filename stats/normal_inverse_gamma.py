from scipy import inf
from scipy.stats import *
from scipy import stats, integrate
import scipy as sp
from scipy.special import gamma
from scipy.misc import derivative
from scipy.stats._multivariate import _process_parameters, _squeeze_output
# from ._multivariate import *
import numpy as np
from numpy import exp
from sys import float_info
from ._multivariate import *

normal_inverse_gamma = None

class normal_inverse_gamma_gen(rv_multivariate_continuous):
    r"""
    A Normal Inverse Gamma (NIG) random variable.
    """

    def __init__(self):
        super().__init__()
        self._mnorm = multivariate_normal
        self._invgamma = invgamma

    def __call__(self, w_0=None, V_0=None, a_0=0, b_0=0):
        return normal_inverse_gamma_frozen(w_0=w_0, V_0=V_0, a_0=a_0, b_0=b_0)

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
        """
        >>> abs(1 - normal_inverse_gamma.cdf([np.inf, np.inf], w_0=np.zeros(1), V_0=np.eye(1), a_0=1, b_0=1)) < 1e-8
        True
        """
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

normal_inverse_gamma = normal_inverse_gamma_gen()

class normal_inverse_gamma_frozen(object):
    def __init__(self, w_0, V_0, a_0, b_0):
        #self.wdim, self.w_0, self.V_0 = _process_parameters(None, w_0, V_0)
        self.w_0 = w_0
        self.V_0 = V_0
        self.wdim = w_0.shape[0]

        self._nig = normal_inverse_gamma_gen()
        self.a_0 = a_0
        self.b_0 = b_0
        self.dim = self.wdim + 1

    def logpdf(self, x):
        return self._nig.logpdf(x, w_0=self.w_0, V_0=self.V_0, a_0=self.a_0, b_0=self.b_0)

    def pdf(self, x):
        return self._nig.pdf(x, w_0=self.w_0, V_0=self.V_0, a_0=self.a_0, b_0=self.b_0)

    def logcdf(self, x):
        return self._nig.logcdf(x, w_0=self.w_0, V_0=self.V_0, a_0=self.a_0, b_0=self.b_0)

    def cdf(self, x):
        return self._nig.cdf(x, w_0=self.w_0, V_0=self.V_0, a_0=self.a_0, b_0=self.b_0)

    def rvs(self, size=1):
        return self._nig.rvs(w_0=self.w_0, V_0=self.V_0, a_0=self.a_0, b_0=self.b_0, size=size)

    def entropy(self):
        return self._nig.entropy(w_0=self.w_0, V_0=self.V_0, a_0=self.a_0, b_0=self.b_0)

# print(normal_inverse_gamma.cdf(np.array([inf, inf]), np.zeros(1), np.ones(1), 1, 1))
