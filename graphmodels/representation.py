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

from .utility import pretty_draw, are_equal_graphs, permutation_dict, compose, lmap, descendants, ancestors
from .factors import TableFactor, DictFactor, FunctionFactor
from .distributions import MathFunction

def reachable(G, source, observed, debug=False, G_reversed=None):
    """
    Finds a set of reachable (in the sense of d-separation) nodes in graph.
    :param G: target graph
    :param source: source node name
    :param observed: a sequence of observed nodes
    :param debug: debug mode on/off
    :param G_reversed: you can provide a graph with reversed edges (for speedup)
    :return: a set of reachable nodes
    """
    V = nx.number_of_nodes(G)
    if G_reversed is None:
        GR = G.reverse()
    else:
        GR = G_reversed
    A = set(sum([list(nx.dfs_preorder_nodes(GR, z)) for z in observed], []))
    Z = observed

    L = [(source, 'up')]
    V = set()
    result = set()
    while len(L) > 0:
        x, d = L.pop()
        if (x, d) in V:
            continue
        if x not in Z:
            result.add((x, d))
        V.add((x, d))
        if d == 'up' and x not in Z:
            for y in G.predecessors_iter(x):
                L.append((y, 'up'))
            for y in G.successors_iter(x):
                L.append((y, 'down'))
        elif d == 'down':
            if x in A:
                for y in G.predecessors_iter(x):
                    L.append((y, 'up'))
            if x not in Z:
                for y in G.successors_iter(x):
                    L.append((y, 'down'))
    result = set([x[0] for x in result])
    if debug:
        pretty_draw(G, node_color = lambda node, attr: '#88DDFF' if node == source else
                   ('#FFFF00' if node in observed else ('#00FF00' if node in result else '#DDDDDD')))
    return result - {source}

def is_covered(G, edge):
    """
    Checks if the edge is covered.
    :param G: graph
    :param edge: edge to check
    :return: is edge covered
    """
    x, y = edge
    return set(G.predecessors_iter(y)) - set(G.predecessors_iter(x)) == {x}

def I_equivalent(G1, G2, debug=False):
    """
    Check if two graphs are I-equivalent
    :param G1: first graph
    :param G2: second graph
    :param debug: debug mode on/off
    :return: are the graphs I-equivalent
    """
    if debug:
        pretty_draw(G1, node_size=lambda *args: 2000)
        pretty_draw(G2, node_size=lambda *args: 2000)

    # same undirected skeleton
    if not are_equal_graphs(G1.to_undirected(), G2.to_undirected()):
        return False

    # same immoralities
    for x in G1.nodes():
        #print(x, G1.predecessors(x))
        for p1, p2 in set.union(set(combinations(G1.predecessors(x), r=2)),
                                set(combinations(G2.predecessors(x), r=2))):
            #print((p1, p2))
            if G1.has_edge(p1, p2) or G1.has_edge(p2, p1):
                continue
            if G2.has_edge(p1, x) and G2.has_edge(p2, x):
                continue
            return False

    # everything OK
    return True

def v_structures(G):
    """
    Iterate over all v-structures in a graph.
    :param G: target graph
    :return: iterator over v-structures in form (node, parent1, parent2)
    """
    for x in G.nodes():
        for p1, p2 in combinations(G.predecessors(x), r=2):
            yield x, p1, p2


def immoralities(G):
    """
    Iterate over all immoralities in a graph.
    :param G: target graph
    :return: iterator over immoralities in form (node, parent1, parent2)
    """
    return filter(lambda v: (not G.has_edge(v[1], v[2])) and (not G.has_edge(v[2], v[1])), v_structures(G))

def unoriented_immoralities(G):
    """
    Iterate over all immoralities in an unoriented graph.
    :param G: target graph
    :return: iterator over immoralities in form (node, parent1, parent2)
    """
    return filter(lambda v: G.has_edge(v[0], v[1]) and G.has_edge(v[0], v[2]) and (not G.has_edge(v[1], v[2])),
        combinations_with_replacement(G.nodes(), 3))

class UGM(nx.Graph):
    def __init__(self):
        super().__init__(self)
        self.potential = permutation_dict()
        self.potential._phook_setitem_ = self._to_factor

    cliques = property(compose(lambda gen: lmap(tuple, gen), nx.find_cliques))

    @property
    def factors(self):
        return list(self.potential.keys())

    def _to_factor(self, key, obj):
        if isinstance(obj, list) or isinstance(obj, np.ndarray):
            return TableFactor(obj, names=key)
        elif isinstance(obj, dict):
            return DictFactor(obj, names=key)
        elif isinstance(obj, MathFunction):
            return FunctionFactor(obj, names=key)
        return obj

    plot = pretty_draw

def random_distr_table(k):
    from numpy.random import dirichlet
    return dirichlet(alpha = tuple([1] * k))

def random_table_factor(k):
    return random_distr_table(2 ** k).reshape(tuple([2] * k))

def random_binary_ugm_distr(G):
    gm = UGM()
    gm.add_nodes_from(G.nodes())
    gm.add_edges_from(G.edges())
    for factor in gm.cliques:
        gm.potential[factor] = random_table_factor(len(factor))
    return gm

def random_binary_ugm(n=10, p=None):
    if p is None: p = log(n) / n
    return random_binary_ugm_distr(nx.erdos_renyi_graph(n=n, p=p))

def UGM_reduce(self, node, val):
    for factor, potential in list(self.potential.items()):
        if node in factor:
            potential.reduce(node, val)
            self.potential.pop(factor)
            factor = list(factor)
            factor.remove(node)
            factor = tuple(factor)
            if len(factor) > 0:
                self.potential[factor] = potential
    super(UGM, self).remove_node(node)

UGM.reduce = UGM_reduce

def factor_graph(G):
    fG = nx.Graph()
    fG.add_nodes_from(G.nodes(), factor=False)
    fG.add_nodes_from(G.factors, factor=True)
    for f in G.factors:
        fG.add_edges_from([(f, var) for var in f])
    return fG

def plot_factor_graph(fG):
    pretty_draw(fG, node_size=lambda node, attr: 1700 if attr['factor'] else 300,
               node_color=lambda node, attr: '#88FF88' if attr['factor'] else '#AAAAFF')

UGM.factor_graph = property(factor_graph)
UGM.fgplot = compose(plot_factor_graph, factor_graph)

def is_I_map(G1, G2, debug=False):
    """
    Is G1 an I-map of G2?
    """
    G2_rev = G2.reverse()
    for x in G1.nodes():
        pa = set(G1.predecessors(x))
        non_descendants = set(G1.nodes()) - descendants(G1, x) - {x}
        if set.intersection(reachable(G2, x, pa, G_reversed=G2_rev), non_descendants - pa):
            if debug:
                print(x, pa, non_descendants - pa, reachable(G2, x, pa, G_reversed=G2_rev))
            return False
    return True

class DGM(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    reachable = reachable
    is_covered = is_covered
    is_I_map = is_I_map
    v_structures = property(v_structures)
    immoralities = property(immoralities)
    plot = pretty_draw

    @property
    def factors(self):
        result = []
        for node, attr in self.nodes(data=True):
            result.append(attr['CPD'])
        return result

def random_binary_dgm(n=10, p=0.2):
    G = nx.gnr_graph(n, p)
    dgm = DGM()
    dgm.add_nodes_from(G.nodes())
    dgm.add_edges_from(G.edges())
    nx.set_node_attributes(dgm, 'CPD', { node: TableFactor(random_table_factor(dgm.in_degree(node) + 1), list(dgm.predecessors(node)) + [node]) for node in dgm.nodes() })
    return dgm

def moralize(G):
    mG = nx.Graph()
    mG.add_nodes_from(G.nodes())
    mG.add_edges_from(G.edges())
    for x, p1, p2 in immoralities(G):
        mG.add_edge(p1, p2)
    return mG

def moral(G):
    return len(list(immoralities(G))) == 0

DGM.moralize = moralize
DGM.moral = property(moral)

from itertools import combinations_with_replacement
def clique_graph(G):
    cg = nx.Graph()
    for clique in nx.find_cliques(G):
        cg.add_node(tuple(clique))
    for node1, node2 in combinations_with_replacement(list(cg.nodes()), 2):
        if set(node1) & set(node2):
            cg.add_edge(node1, node2)
    return cg

def ugm_from_factors(factors):
    ugm = UGM()
    for factor, potential in factors.items():
        ugm.add_nodes_from(factor)
        ugm.add_edges_from(combinations(factor, 2))
        ugm.potential[tuple(factor)] = potential
    return ugm

def graph_from_factors(factors):
    ugm = nx.Graph()
    for factor in factors:
        ugm.add_nodes_from(factor)
        ugm.add_edges_from(combinations(factor, 2))
    return ugm