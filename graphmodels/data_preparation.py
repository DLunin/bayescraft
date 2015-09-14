import numpy as np
import pandas as pd
from .utility import *


def is_discrete(data):
    return all(map(lambda x: float(x).is_integer(), data))

def discretize(data, bins, force=True, exclude=None):
    """
    Binning continuous data array to get discrete data array.
    :param data: target numpy array
    :return: discretized array
    """
    if exclude:
        return data
    if not is_discrete(data) or force:
        sdata = np.sort(data)
        borders = np.zeros((bins + 1,))
        for i in range(bins):
            borders[i] = sdata[int((i / float(bins)) * len(data))]
        borders[-1] = np.inf
        borders[0] = -np.inf
        assert len(borders) == bins + 1
        result = np.zeros(data.shape)
        for i in range(len(result)):
            counter = 0
            while data[i] > borders[counter]:
                counter += 1
            result[i] = counter-1
            assert result[1] < bins
        return result
    else: return data

def pandas_discretize(data, bins, all=False, exclude=set()):
    dv = np.transpose(np.asarray(lmap(lambda x: discretize(x[0], bins, force=all, exclude=x[1] in exclude), zip(np.transpose(data.values), data.columns))))
    result = pd.DataFrame(data=dv, columns=data.columns.values)
    return result

class DataPreparer:
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for var in data.columns:
            data[var] = self._prepare_single(data[var].values)
        return data

class DiscreteDataPreparer(DataPreparer):
    def __init__(self, max_n: int, discr_n: int):
        self.max_n = max_n
        self.discr_n = discr_n

    def _prepare_single(self, arr) -> np.ndarray:
        if is_discrete(arr):
            if np.max(arr) - np.min(arr) >= self.max_n:
                return discretize(arr, self.discr_n)
            else:
                return arr - np.array([np.min(arr)] * arr.shape[0])
        else:
            return discretize(arr, self.discr_n)

