from copy import deepcopy
from itertools import combinations

from .utility import *
from .representation import UGM, DGM

def dgm_reduce(factors, name, val):
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

def sum_product(factors, query, ordering, evidence=None, build_induced_graph=False):
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
        factors = dgm_reduce(factors, key, val)

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

        factor_product.marginalize(var)
        if not factor_product.empty():
            factors.add(factor_product)
    result = factors.pop()
    for i, factor in enumerate(factors):
        assert set(factor.names) <= set(query)
        result = result * factor

    if build_induced_graph:
        assert nx.is_chordal(IG)
        return result, IG
    return result

def eliminate_variable(factors, x):
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

def max_cardinality(G):
    """
    Max-cardinality variable ordering.
    :param G: target graph
    :return: node ordering -- a list of nodes
    """
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
    return ranking