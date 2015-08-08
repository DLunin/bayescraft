import networkx as nx
import numpy as np
from numpy import log, exp
import scipy as sp
import pymc
from scipy import stats
import emcee
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mutual_info_score
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from itertools import *
from NPEET import mi
from math import sqrt
import math
from copy import deepcopy
from itertools import combinations

from .utility import *
from .representation import UGM, DGM

def dgm_reduce(factors, name, val):
    result = []
    for factor in factors:
        fcopy = deepcopy(factor)
        fcopy.reduce(name, val)
        result.append(fcopy)
    return result

def sum_product(factors, query, ordering, evidence=None, build_induced_graph=False):
    if evidence is None:
        evidence = dict()

    for key, val in evidence.items():
        factors = dgm_reduce(factors, key, val)

    if build_induced_graph:
        IG = UGM()

    factors = set(factors)
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
    for factor in factors:
        result = result * factor

    if build_induced_graph:
        assert nx.is_chordal(IG)
        return result, IG
    return result

def eliminate_variable(factors, x):
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