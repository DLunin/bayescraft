from numpy import log, exp
import itertools

from .utility import pretty_print_distr_dict, pretty_print_distr_table, pretty_draw, lmap
from .distributions import *

class TableFactor:
    """
    A factor repesented by table, that is, a multidimensional array; indexes represent
    arguments while elements represent values.

    Supports only discrete argument values. Moreover, they must be in range 0-N, that is,
    consecutive integers.
    """
    def __init__(self, table, names):
        self.table = np.array(table)
        self.names = list(names)

    def __call__(self, **kwargs):
        args = tuple([kwargs[name] for name in self.names])
        try:
            return self.table[args]
        except (IndexError, KeyError):
            return 0

    def _repr_html_(self):
        return pretty_print_distr_table(self.table, names=self.names)._repr_html_()

    def energy(self, **kwargs):
        """
        :param kwargs: argument values
        :return: negative log value of factor
        """
        return -log(self.__call__(*kwargs))

    def reduce(self, var_name, val):
        """
        Reduce a variable in factor
        :param var_name: name of reduced variable
        :param val: value to assign to reduced variable
        :return: None
        """
        if var_name not in self.names:
            return None
        var = self.names.index(var_name)
        self.table = self.table[tuple([slice(None) if i != var else val for i in range(len(self.table.shape))])]
        self.names.pop(var)

    def marginalize(self, var_name):
        """
        Marginalize a variable.
        :param var_name: name of variable to be marginalized
        :return: None
        """
        if var_name not in self.names:
            return None
        var = self.names.index(var_name)
        self.table = np.sum(self.table, axis=var)
        self.names.pop(var)

    def __mul__(self, other):
        """
        Factor product.
        :param other: right-hand side factor
        :return: the product of factors
        """
        assert isinstance(other, TableFactor)
        common_vars = set.intersection(set(self.names), set(other.names))
        self_vars = list(set(self.names) - common_vars)
        other_vars = list(set(other.names) - common_vars)
        common_vars = list(common_vars)

        common_vars_idx_self = lmap(self.names.index, common_vars)
        common_vars_idx_other = lmap(other.names.index, common_vars)
        self_vars_idx = lmap(self.names.index, self_vars)
        other_vars_idx = lmap(other.names.index, other_vars)

        shape = [self.table.shape[common_vars_idx_self[var]] for var in range(len(common_vars))] + \
            [self.table.shape[self_vars_idx[var]] for var in range(len(self_vars))] + \
            [other.table.shape[other_vars_idx[var]] for var in range(len(other_vars))]

        result = np.zeros(tuple(shape))
        for idx in itertools.product(*map(range, shape)):
            common_part = idx[:len(common_vars)]
            self_part = idx[len(common_vars):len(common_vars) + len(self_vars)]
            other_part = idx[len(common_vars) + len(self_vars):]

            self_arg = [0] * (len(common_part) + len(self_part))
            other_arg = [0] * (len(common_part) + len(other_part))

            for idx_self, idx_other, val in zip(common_vars_idx_self, common_vars_idx_other, common_part):
                self_arg[idx_self] = val
                other_arg[idx_other] = val

            for idx_self, val in zip(self_vars_idx, self_part):
                self_arg[idx_self] = val

            for idx_self, val in zip(other_vars_idx, other_part):
                other_arg[idx_self] = val

            result[tuple(idx)] = self.table[tuple(self_arg)] * other.table[tuple(other_arg)]
        return TableFactor(result, names = common_vars + self_vars + other_vars)

    @property
    def dim(self):
        """
        :return: number of arguments of the factor
        """
        return len(self.table.shape)

    @property
    def domain(self):
        """
        :return: domain of the factor
        """
        return ProductDomain(*[DiscreteDomain(list(range(dim))) for dim in self.table.shape])

    def __contains__(self, name):
        """
        :param name: variable name
        :return: do the factor arguments contain that variable
        """
        return name in self.names

    def empty(self):
        """
        :return: is the factor empty; the factor is empty if it has 0 variables as arguments
        """
        return len(self.names) == 0


class DictFactor:
    """
    A factor represented as dict; keys are argument values and values are factor values.
    Keys are stored as tuples.
    """
    def __init__(self, d, names):
        self.dict = d
        self.names = list(names)
        self.dim = len(self.names)

    def __call__(self, **kwargs):
        args = tuple([kwargs[name] for name in self.names])
        try:
            return self.dict[args]
        except KeyError:
            return 0

    def energy(self, **kwargs):
        """
        :param kwargs: argument values
        :return: negative log value of factor
        """
        return -log(self.__call__(**kwargs))

    def _repr_html_(self):
        return pretty_print_distr_dict(self.dict, names=self.names)._repr_html_()

    @property
    def domain(self):
        values = [set() for var in self.dict]


class FunctionFactor:
    """
    Factor represented as a Python function.
    """
    def __init__(self, f, names):
        self.f = f
        self.names = list(names)
        self.dim = len(self.names)

    def __call__(self, **kwargs):
        args = tuple([kwargs[name] for name in self.names])
        if self.dim is None:
            self.dim = len(args)
        return self.f(*args)

    def energy(self, **kwargs):
        """
        :param kwargs: argument values
        :return: negative log value of factor
        """
        return -log(self.f(**kwargs))

    def reduce(self, var_name, val):
        """
        Reduce a variable in factor
        :param var_name: name of reduced variable
        :param val: value to assign to reduced variable
        :return: None
        """
        if var_name not in self.names:
            return None
        var = self.names.index(var_name)

        def f(*args):
            args.insert(var, val)
            return self.f(*args)
        self.f = f
        self.names.pop(var)

    def marginalize(self, var_name):
        """
        Marginalize a variable.
        :param var_name: name of variable to be marginalized
        :return: None
        """
        var = self.names.index(var_name)
        self.f = self.f.integrate_along(var)
        self.dom
        self.names.pop(var)

    @property
    def domain(self):
        """
        :return: domain of the factor
        """
        return self.f.domain

    def __contains__(self, name):
        """
        :param name: variable name
        :return: do the factor arguments contain that variable
        """
        return name in self.names