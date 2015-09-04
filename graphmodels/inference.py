from copy import deepcopy
from itertools import combinations

from .utility import *
from .representation import UGM, DGM
from .factors import Factor

class Inferencer:
    def fit(self, graphical_model, **options):
        raise NotImplementedError()

    def prob(self, variables: list, observed: dict) -> float:
        raise NotImplementedError()

def complete_ordering(G, ordering):
    result = ordering
    for node in set.difference(set(G.nodes()), set(ordering)):
        result.append(node)
    return result

class SumProductInferencer:
    def __init__(self):
        self.graphical_model = DGM()
        self.variables = []
        self.ordering = None

    def fit(self, graphical_model, **options):
        self.graphical_model = graphical_model
        self.variables = list(self.graphical_model.nodes())
        self.ordering = self.max_cardinality()

    def _prob_single(self, variables: list, observed: dict, **options) -> Factor:
        ordering = options['ordering'] if 'ordering' in options else self.ordering
        return SumProductInferencer._sum_product(self.graphical_model.factors(),
                                                 variables,
                                                 ordering,
                                                 evidence=observed)

    def prob(self, variables: list, observed: pd.DataFrame, **options) -> [Factor]:
        result = []
        for point in observed.values:
            arg = dict(zip(observed.columns.values, point))
            result.append(self._prob_single(variables, arg, **options))
        return result

    def _predict_single(self, variables: list, observed: dict, **options) -> dict:
        return self._prob_single(variables, observed, **options).argmax_p()

    def predict(self, variables: list, observed: pd.DataFrame, **options) -> pd.DataFrame:
        dicts_list = []
        for point in observed.values:
            arg = dict(zip(observed.columns.values, point))
            dicts_list.append(self._predict_single(variables, arg, **options))
        return pd.DataFrame(dicts_list)

    @staticmethod
    def _dgm_reduce(factors, name, val):
        """
        Reduce each factor in a sequence of factors.
        :param factors: the seqence of factors
        :param name: name of the variable to be reduced
        :param val: value assigned to the reduced variable
        :return: a list of reduced factors
        """
        result = []
        for factor in factors:
            fcopy = deepcopy(factor)
            fcopy.reduce(name, val)
            result.append(fcopy)
        return result

    @staticmethod
    def _sum_product(factors, query, ordering, evidence=None, build_induced_graph=False):
        """
        Sum-product inference algorithm.
        :param factors: a sequence of factors
        :param query: remaining nodes after marginalization
        :param ordering: an ordering of variables
        :param evidence: observed nodes
        :param build_induced_graph:
        :return: a sequence of factors with all but `query` variables marginalized
        """
        from copy import deepcopy
        ordering = list(filter(lambda x: x not in query, ordering))

        if evidence is None:
            evidence = dict()

        for key, val in evidence.items():
            factors = SumProductInferencer._dgm_reduce(factors, key, val)

        if build_induced_graph:
            IG = UGM()

        factors = set(deepcopy(factors))
        for var in ordering:
            factor_product = None
            for factor in list(factors):
                if var in factor:
                    factor_product = (factor_product * factor) if factor_product is not None else factor
                    factors.remove(factor)

            if build_induced_graph:
                IG.add_nodes_from(factor_product.names)
                IG.add_edges_from(combinations(factor_product.names, 2))
                IG.potential[tuple(factor_product.names)] = factor_product

            if factor_product is not None:
                factor_product.marginalize(var)
                if not factor_product.empty():
                    factors.add(factor_product)

        result = factors.pop()
        for i, factor in enumerate(factors):
            result = result * factor

        if build_induced_graph:
            assert nx.is_chordal(IG)
            return result, IG
        return result

    @staticmethod
    def _eliminate_variable(factors, x):
        """
        Eliminate variable from a sequence of factors
        :param factors: a target sequence of factors
        :param x: variable to be eliminated
        :return: list of factors with the variable eliminated, in form of [ { node, ... } ]
        """
        factors = set(map(tuple, factors))
        factor_product = set()
        for factor in list(factors):
            if x in factor:
                factor_product = factor_product | set(factor)
                factors.remove(factor)
        factor_product.remove(x)
        if factor_product:
            factors.add(tuple(factor_product))
        return lmap(set, factors)

    def max_cardinality(self):
        """
        Max-cardinality variable ordering.
        :param G: target graph
        :return: node ordering -- a list of nodes
        """
        G = self.graphical_model

        marked = set()
        not_marked = set(G.nodes())

        def score(node):
            result = 0
            for neigh in G.neighbors_iter(node):
                if neigh in marked:
                    result += 1
            return result

        ranking = []
        while not_marked:
            x = max(not_marked, key=score)
            ranking.append(x)
            marked.add(x)
            not_marked.remove(x)
        return ranking[::-1]
