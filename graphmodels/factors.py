import numpy.random as rand
from numpy import log, exp
import itertools
from itertools import product
import pandas as pd

from .utility import pretty_print_distr_dict, pretty_print_distr_table, pretty_draw, lmap, plot_distr
from .distributions import *

class Factor:
    def __call__(self, *args, kwargs):
        return 0.

    def energy(self, kwargs):
        """
        :param kwargs: argument values
        :return: negative log value of factor
        """
        return -log(self.__call__(kwargs))

class TableFactor(Factor):
    """
    A factor repesented by table, that is, a multidimensional array; indexes represent
    arguments while elements represent values.

    Supports only discrete argument values. Moreover, they must be in range 0-N, that is,
    consecutive integers.
    """
    def __init__(self, table, names):
        super().__init__()
        self.table = np.array(table)
        self.names = names

    def factor(self):
        return self

    def __call__(self, kwargs):
        args = tuple([kwargs[name] for name in self.names])
        try:
            return self.table[args]
        except (IndexError, KeyError):
            return 0

    def _repr_html_(self):
        return pretty_print_distr_table(self.table, names=self.names)._repr_html_()

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

    @property
    def n_parameters(self):
        """
        Number of parameters which specify the factor.
        """
        result = 1
        for n in self.table.shape:
            result *= n
        return result

    @staticmethod
    def mle(data):
        headers = list(data.columns)
        data = data.values
        values = lmap(set, data.T)
        shape = tuple(map(len, values))
        table = np.zeros(*shape)
        for point in data:
            table[tuple(point)] += 1
        return TableFactor(table, headers)

    def sample(self, n_samples):
        """
        Sample data from the factor.
        """
        result = []
        values = [self.table.shape[self.names.index(name)] for name in self.names]
        norm_const = np.sum(np.ravel(self.table))
        rands = rand.rand(n_samples) * norm_const
        for sample in range(n_samples):
            current = 0
            for assignment in product(*lmap(range, values)):
                current += self(assignment)
                if current > rands[sample]:
                    result.append(assignment)
                    break
        result = np.array(result)
        return pd.DataFrame(data=result, columns=self.names)

class TableCPD(TableFactor):
    """
    A CPD repesented by table, that is, a multidimensional array; indexes represent
    arguments while elements represent values.

    Supports only discrete argument values. Moreover, they must be in range 0-N, that is,
    consecutive integers.

    The difference between CPD and Factor classes is that values of CPD must sum to 1.
    """
    def __init__(self, table, var_names, cond_names):
        super().__init__(table, cond_names + var_names)
        self.var_names = var_names
        self.cond_names = cond_names
        self._renormalize()

    @property
    def var_dim(self):
        return len(self.var_names)

    @property
    def cond_dim(self):
        return len(self.cond_names)

    def _renormalize(self):
        for x in product(*lmap(range, self.table.shape[:self.cond_dim])):
            constant = np.sum(self.table[tuple(x)])
            if constant > 0:
                self.table[tuple(x)] = self.table[tuple(x)] / constant
            else:
                self.table[tuple(x)] = 0

    def reduce(self, var_name, val):
        super().reduce(var_name, val)
        if var_name in self.var_names:
            self._renormalize()

    @property
    def n_parameters(self):
        """
        Number of parameters which specify the CPD.
        """
        result = 1
        for n in self.table.shape:
            result *= n
        return result - self.var_dim

    def sample(self, n_samples, observed=None):
        """
        Sample data from the CPD.
        """
        if observed is None:
            observed = pd.DataFrame()
        nonobserved = list(filter(lambda x: x not in observed, self.names))
        result = { name : [] for name in nonobserved }
        values = [self.table.shape[self.names.index(name)] for name in nonobserved]
        rands = rand.rand(n_samples)
        for sample in range(n_samples):
            current = 0
            for assignment in product(*lmap(range, values)):
                assignment_dict = { name : assignment[i] for i, name in enumerate(nonobserved) }
                assignment_dict.update({key : observed[key][sample] for key in observed.columns })
                current += self(assignment_dict)

            current_max = current
            current = 0
            for assignment in product(*lmap(range, values)):
                assignment_dict = { name : assignment[i] for i, name in enumerate(nonobserved) }
                assignment_dict.update({key : observed[key][sample] for key in observed.columns })
                current += self(assignment_dict)
                if current > rands[sample] * current_max:
                    for key, val in zip(nonobserved, assignment):
                        result[key].append(val)
                    break
        return pd.DataFrame(result)

    @staticmethod
    def mle(data, conditioned=None):
        """
        Maximum Likelihood Estimation (MLE).
        """
        if conditioned is None:
            conditioned = []
        headers = list(data.columns)
        data = data.values
        values = lmap(max, data.T)
        shape = tuple(map(lambda x: x + 1, values))
        table = np.zeros(shape)
        for point in data:
            table[tuple(point)] += 1
        result = TableCPD(table, [name for name in headers if name not in conditioned], conditioned)
        return result

    def factor(self):
        return TableFactor(self.table, self.names)

class DictFactor(Factor):
    """
    A factor represented as dict; keys are argument values and values are factor values.
    Keys are stored as tuples.
    """
    def __init__(self, d, names):
        self.dict = d
        self.names = list(names)
        self.dim = len(self.names)

    def __call__(self, kwargs):
        args = tuple([kwargs[name] for name in self.names])
        try:
            return self.dict[args]
        except KeyError:
            return 0

    def _repr_html_(self):
        return pretty_print_distr_dict(self.dict, names=self.names)._repr_html_()


class FunctionFactor(Factor):
    """
    Factor represented as a Python function.
    """
    def __init__(self, f, names):
        self.f = f
        self.names = list(names)

    @property
    def dim(self):
        return len(self.names)

    def __call__(self, kwargs):
        args = tuple([kwargs[name] for name in self.names])
        return self.f(*args)

    def factor(self):
        return self

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


class ParametricFunctionFactor(Factor):
    """
    Factor represented as a Python function.
    """
    def __init__(self, f, names, **params):
        self.f = f
        self.names = list(names)
        self.params = params

    @property
    def dim(self):
        return len(self.names)

    def __call__(self, kwargs):
        args = tuple([kwargs[name] for name in self.names])
        return self.f(*args, **self.params)

    def factor(self):
        return self

    def __mul__(self, other):
        y = set(self.names).intersection(set(other.names))
        x = list(set(self.names) - y)
        z = list(set(other.names) - y)
        x_idx = [i for i, x in enumerate(self.names) if x not in y]
        y_idx_self = [i for i, x in enumerate(self.names) if x in y]
        y_idx_other = [i for i, x in enumerate(other.names) if x in y]
        z_idx = [i for i, z in enumerate(other.names) if z not in y]
        other_ext_f = other.f.extended(len(x))
        return ParametricFunctionFactor(self.f.reordered(x_idx + y_idx_self).extended(len(z)) * \
                                        other_ext_f.reordered(list(range(len(z) + len(y), other_ext_f.dim)) + \
                                                              y_idx_other + z_idx), x + list(y) + z)

    def reduce(self, var_name, val):
        """
        Reduce a variable in factor
        :param var_name: name of reduced variable
        :param val: value to assign to reduced variable
        :return: None
        """
        idx = self.names.index(var_name)
        self.f.reduce(idx, val)

    def marginalize(self, var_name):
        """
        Marginalize a variable.
        :param var_name: name of variable to be marginalized
        :return: None
        """
        var = self.names.index(var_name)
        self.f = self.f.marginalize(var)
        self.names.pop(var)

    @property
    def domain(self):
        """
        :return: domain of the factor
        """
        return self.f.domain

    @property
    def n_parameters(self):
        """
        Number of parameters which specify the factor.
        """
        def count_scalars(x):
            try:
                return sum(map(count_scalars, list(x)))
            except TypeError:
                return 1

        result = 0
        for pname, param in self.params.items():
            result += count_scalars(param)
        return result

    def __contains__(self, name):
        """
        :param name: variable name
        :return: do the factor arguments contain that variable
        """
        return name in self.names

    def sample(self, n_samples):
        return pd.DataFrame(data=self.f.rvs(size=n_samples), columns=self.names)

    @staticmethod
    def mle(data, model):
        header = list(data.columns)
        f = model.mle(data.values)
        return ParametricFunctionFactor(f, header)

    def empty(self):
        return len(self.names) == 0

class ParametricFunctionCPD(ParametricFunctionFactor):
    """
    CPD represented as a Python function.
    """
    def __init__(self, f, var_names, cond_names):
        super().__init__(f, var_names + cond_names)

    def factor(self):
        return ParametricFunctionFactor(self.f, self.names)

    def reduce(self, var_name, val):
        """
        Reduce a variable in factor
        :param var_name: name of reduced variable
        :param val: value to assign to reduced variable
        :return: None
        """
        idx = self.names.index(var_name)
        self.f.reduce(idx, val)

    def sample(self, n_samples, observed=None):
        if observed is None:
            observed = pd.DataFrame()
        observed_names = list(observed.columns)
        non_observed_names = [name for name in self.names if name not in observed_names]
        result = []
        for sample_i in range(n_samples):
            assignment = [observed[name][sample_i] if name in observed_names else None for name in self.names]
            result.append(np.atleast_1d(self.f.reduce(assignment).rvs()))
        result = np.array(result)
        return pd.DataFrame(data=result, columns=non_observed_names)

    @staticmethod
    def mle(data, model, conditioned=None):
        if conditioned is None:
            conditioned = []
        header = list(data.columns)
        variables = [name for name in header if name not in conditioned]
        dataval = data[variables + conditioned].values
        f = model.mle(dataval)
        return ParametricFunctionCPD(f, variables, conditioned)

    def plot(self):
        plot_distr(self.f.pdf)


class ParametricFunctionModel:
    def __init__(self, model):
        self.model = model

    def mle(self, data, conditioned=None):
        return ParametricFunctionCPD.mle(data, self.model, conditioned=conditioned)