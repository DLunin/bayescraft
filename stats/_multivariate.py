import numpy as np
import scipy as sp
from scipy.misc import derivative

class rv_multivariate_continuous:
    def __init__(self):
        pass

    def pdf(self, x, **params):
        return derivative(self.cdf(x, **params))

    def logcdf(self, *args, **kwargs):
        return np.log(self.cdf(*args, **kwargs))

    def logcdf(self, *args, **kwargs):
        return np.log(self.cdf(*args, **kwargs))