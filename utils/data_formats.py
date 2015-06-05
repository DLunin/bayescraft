import numpy as np
import numpy.linalg as lin

def to_column_vector(v):
    if isinstance(v, np.ndarray):
        assert len(v.shape) <= 2
        if len(v.shape) == 2:
            assert v.shape[0] == 1 or v.shape[1] == 1
            if v.shape[0] == 1:
                return np.matrix(v).T
            else:
                return np.matrix(v)
        else:
            return np.matrix(v).T
    elif isinstance(v, np.matrix):
        if v.shape[1] != 1:
            assert v.shape[0] == 1
            return v.T
        return v
    else:
        return to_column_vector(np.matrix(v))


def is_column_vector(v):
    return isinstance(v, np.matrix) and v.shape[1] == 1

def is_row_vector(v):
    return isinstance(v, np.matrix) and v.shape[0] == 1

def to_1d_array(v):
    if isinstance(v, np.ndarray):
        assert sum(v.shape) - len(v.shape) + 1 == max(v.shape)
        return np.ravel(v)
    elif isinstance(v, np.matrix):
        assert v.shape[0] == 1 or v.shape[1] == 1
        return np.ravel(v)
    else:
        return to_1d_array(np.array(v))

def to_matrix(m):
    if isinstance(m, np.ndarray):
        assert len(m.shape) <= 2
        if len(m.shape) == 1:
            return np.matrix(m).T
        else:
            return np.matrix(m)
    else:
        return to_matrix(np.array(m))

def to_scalar(v):
    v = np.ravel(v)
    assert v.shape[0] == 1
    return float(v[0])