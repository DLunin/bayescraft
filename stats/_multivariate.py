import numpy as np

class rv_multivariate_continuous:
    def logcdf(self, *args, **kwargs):
        return np.log(self.cdf(*args, **kwargs))