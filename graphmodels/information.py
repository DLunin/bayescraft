from scipy import stats
from scipy.stats import gaussian_kde
#from NPEET import mi
from numpy import log, exp

from .utility import *

def monte_carlo_integration(distr_rvs, f, initial_chunk_size=100, chunk_size=10, eps=0.1):
    """
    Monte-Carlo numerical integration.
    :param distr_rvs: random sampler (takes size as `size` parameter and returns numpy array of samples
    :param f: function to be integrated
    :param initial_chunk_size: initial chunk size
    :param chunk_size: chunk size
    :param eps: precision
    :return: integral value
    """
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
    """
    Kernel Density Estimator (KDE)
    """
    def __init__(self, data, h=1):
        """
        :param h: window size
        """
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
        """
        :return: estimated density
        """
        return sum(map(lambda distr: distr.pdf(x), self.distrs)) / self.n / self.h


mutinfo_density_estimator = lambda data: gaussian_kde(data)
def mutual_information_naive(data1, data2, debug=False):
    """
    Mutual information estimator based on approximating the distibutions and numerical integration
    using Monte-Carlo method.
    :param data1: first distribution data (rows -- samples, cols -- axes)
    :param data2: second distribution data (rows -- samples, cols -- axes)
    :param debug: debug mode on/off
    :return: estimated mutual information
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

    return monte_carlo_integration(lambda size=1: np.transpose(distr12.resample(size=size)), f)[0]

#mutual_information = lambda *args, debug=None, **kwargs: mi(*args, **kwargs)
mutual_information = mutual_information_naive

def pairwise_mutual_info(data, v1, v2):
    """
    Mutual information between two variables.
    :param data: dataset containing the variables
    :param v1: name (index) of the first variable
    :param v2: name (index) of the second variable
    :return: estimated mutual information
    """
    return mutual_information(data[:, v1:v1+1], data[:, v2:v2+1])

def conditional_mutual_information(data, x, y, z, bw_method=None):
    """
    Conditional mutual information estimator.
    :param x: variables of the first distribution (list)
    :param y: conditioned variables (list)
    :param z: variables of the second distribution (list)
    :param bw_method: parameter of gaussian_kde of scipy
    :return: estimated conditional mutual information
    """
    n_x, n_y, n_z = len(x), len(y), len(z)
    assert n_x == 1

    if len(y) == 0:
        data_x = data[:, x]
        data_z = data[:, z]
        return mutual_information(data_x, data_z)

    y_distr = gaussian_kde(np.transpose(data[:, y]), bw_method=bw_method)
    xy_distr = gaussian_kde(np.transpose(data[:, x + y]), bw_method=bw_method)
    xyz_distr = gaussian_kde(np.transpose(data[:, x + y + z]), bw_method=bw_method)
    z_distr = gaussian_kde(np.transpose(data[:, z]), bw_method=bw_method)
    yz_distr = gaussian_kde(np.transpose(data[:, y + z]), bw_method=bw_method)

    def f(s):
        x_part = s[:n_x]
        y_part = s[n_x:n_y+n_x]
        xy_part = s[:n_x + n_y]
        yz_part = s[n_x:n_x+n_y+n_z]
        z_part = s[n_x+n_y:n_x+n_y+n_z]
        return log(xyz_distr(s)) + log(y_distr(y_part)) - log(xy_distr(xy_part)) - log(yz_distr(yz_part))

    return monte_carlo_integration(lambda size=1: np.transpose(xyz_distr.resample(size=size)), f)

def conditional_mutual_information2(data_x, data_y, data_z, bw_method=None):
    """
    Conditional mutual information estimator.
    :param data_x: first distribution data
    :param data_y: conditioned variables data
    :param data_z: second distribution data
    :param bw_method: parameter of gaussian_kde of scipy
    :return: estimated conditional mutual information
    """
    n_x, n_y, n_z = data_x.shape[1], data_y.shape[1], data_z.shape[1]
    assert n_x == 1

    if data_y.shape[0] == 0:
        return mutual_information(data_x, data_z)
    y_distr = gaussian_kde(np.transpose(data_y), bw_method=bw_method)
    xy_distr = gaussian_kde(np.transpose(np.hstack([data_x, data_y])), bw_method=bw_method)
    xyz_distr = gaussian_kde(np.transpose(np.hstack([data_x, data_y, data_z])), bw_method=bw_method)
    z_distr = gaussian_kde(np.transpose(data_z), bw_method=bw_method)
    yz_distr = gaussian_kde(np.transpose(np.hstack([data_y, data_z])), bw_method=bw_method)

    def f(s):
        x_part = s[:n_x]
        y_part = s[n_x:n_y+n_x]
        xy_part = s[:n_x + n_y]
        yz_part = s[n_x:n_x+n_y+n_z]
        z_part = s[n_x+n_y:n_x+n_y+n_z]
        return log(xyz_distr(s)) + log(y_distr(y_part)) - log(xy_distr(xy_part)) - log(yz_distr(yz_part))

    return monte_carlo_integration(lambda size=1: np.transpose(xyz_distr.resample(size=size)), f)

def discrete_distribution(data):
    """
    Build a discrete distribution in form of { point : probability }
    :param data: numpy.array, (data points, dimension)
    :return: dict of distribution
    """
    distr = { }
    n = data.shape[0]
    for point in data:
        if tuple(point) in distr:
            distr[tuple(point)] += 1. / n
        else:
            distr[tuple(point)] = 1. / n
    return distr

def discrete_mutual_information(data_x, data_y):
    """
    Discrete mutual information.
    :param data_x: first distribution data
    :param data_y: second distribution data
    :return: mutual information
    """
    x_dim = data_x.shape[1]
    distr_x = discrete_distribution(data_x)
    distr_y = discrete_distribution(data_y)
    distr_xy = discrete_distribution(np.hstack([data_x, data_y]))
    result = 0.
    for point, p_xy in distr_xy.items():
        x_part = point[:x_dim]
        y_part = point[x_dim:]
        p_x = distr_x[tuple(x_part)]
        p_y = distr_y[tuple(y_part)]
        result += p_xy * (log(p_xy) - log(p_x) - log(p_y))
    return result