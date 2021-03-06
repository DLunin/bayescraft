import networkx as nx
import numpy as np
from numpy import log, exp
from itertools import *
import pandas as pd
from types import GeneratorType

from .utility import *
from .factors import *
from .distributions import *
from bayescraft.framework import *

class DGM(nx.DiGraph):
    """
    Directed Graphical Model
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cpd = { }
        self.model = { }

    def factors(self):
        result = []
        for key, val in self.cpd.items():
            result.append(val.factor())
        return result

    def reachable(self, source, observed, debug: bool=False) -> set:
        """
            Finds a set of reachable (in the sense of d-separation) nodes in graph.
            :param self: target graph
            :param source: source node name
            :param observed: a sequence of observed nodes
            :param debug: debug mode on/off
            :return: a set of reachable nodes
            """
        V = nx.number_of_nodes(self)
        A = set(sum([list(nx.dfs_preorder_nodes(self.reverse(), z)) for z in observed], []))
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
                for y in self.predecessors_iter(x):
                    L.append((y, 'up'))
                for y in self.successors_iter(x):
                    L.append((y, 'down'))
            elif d == 'down':
                if x in A:
                    for y in self.predecessors_iter(x):
                        L.append((y, 'up'))
                if x not in Z:
                    for y in self.successors_iter(x):
                        L.append((y, 'down'))
        result = set([x[0] for x in result])
        if debug:
            pretty_draw(self, node_color=lambda node, attr: '#88DDFF' if node == source else
            ('#FFFF00' if node in observed else ('#00FF00' if node in result else '#DDDDDD')))
        return result - {source}

    def is_covered(self, edge) -> bool:
        """
        Checks if the edge is covered.
        :param self: graph
        :param edge: edge to check
        :return: is edge covered
        """
        x, y = edge
        return set(self.predecessors_iter(y)) - set(self.predecessors_iter(x)) == {x}

    def is_I_map(self, other, debug: bool=False) -> bool:
        """
        Is G1 an I-map of G2?
        """
        G2_rev = other.reverse()
        for x in self.nodes():
            pa = set(self.predecessors(x))
            non_descendants = set(self.nodes()) - descendants(self, x) - {x}
            if set.intersection(other.reachable(other, x, pa, G_reversed=G2_rev), non_descendants - pa):
                if debug:
                    print(x, pa, non_descendants - pa, self.reachable(x, pa))
                return False
        return True

    def is_I_equivalent(self, other, debug: bool=False) -> bool:
        """
        Check if two graphs are I-equivalent
        :param self: first graph
        :param other: second graph
        :param debug: debug mode on/off
        :return: are the graphs I-equivalent
        """
        if debug:
            pretty_draw(self, node_size=lambda *args: 2000)
            pretty_draw(other, node_size=lambda *args: 2000)

        # same undirected skeleton
        if not are_equal_graphs(self.to_undirected(), other.to_undirected()):
            return False

        # same immoralities
        for x in self.nodes():
            #print(x, G1.predecessors(x))
            for p1, p2 in set.union(set(combinations(self.predecessors(x), r=2)),
                                    set(combinations(other.predecessors(x), r=2))):
                #print((p1, p2))
                if self.has_edge(p1, p2) or self.has_edge(p2, p1):
                    continue
                if other.has_edge(p1, x) and other.has_edge(p2, x):
                    continue
                return False

        # everything OK
        return True

    @property
    def is_moral(self):
        """
        A graph is moral if it has no immoralities.
        :param self: target graph
        :return: is target graph moral
        """
        return len(list(self.immoralities)) == 0

    @property
    def immoralities(G):
        """
        Iterate over all immoralities in a graph.
        :param G: target graph
        :return: iterator over immoralities in form (node, parent1, parent2)
        """
        return filter(lambda v: (not G.has_edge(v[1], v[2])) and (not G.has_edge(v[2], v[1])), G.v_structures)

    @property
    def v_structures(G) -> GeneratorType:
        """
        Iterate over all v-structures in a graph.
        :param G: target graph
        :return: iterator over v-structures in form (node, parent1, parent2)
        """
        for x in G.nodes():
            for p1, p2 in combinations(G.predecessors(x), r=2):
                yield x, p1, p2

    def sample(self, n_samples) -> pd.DataFrame:
        """
        Sample data from DGM.
        """
        result = pd.DataFrame()
        for node in nx.topological_sort(self):
            parents = list(self.predecessors(node))
            assignment = result[parents]
            data_part = self.cpd[node].sample(n_samples=n_samples, observed=assignment)
            result = pd.concat([result, data_part], axis=1)
        return result

    def mle(self, data: pd.DataFrame, values: dict=None) -> dict:
        cpd = { }
        for node in nx.topological_sort(self):
            parents = list(self.predecessors(node))
            if values is not None:
                current_values = [values[x] for x in parents + [node]]
            else:
                current_values = None
            cpd[node] = self.model[node].mle(data[parents + [node]], conditioned=parents, values=current_values)
        return cpd

    plot = pretty_draw

    # @property
    # def factors(self):
    #     result = []
    #     for node, attr in self.nodes(data=True):
    #         result.append(attr['CPD'])
    #     return result

    @staticmethod
    def from_data(data, discretizing_bins=None):
        """
        Initialize DGM from pandas DataFrame.
        Add nodes, set some of their attributes(data, |Val|).
        :param data: pandas DataFrame
        :return: DGM
        """
        G = DGM()
        G.add_nodes_from(data.columns.values)
        if discretizing_bins:
            nx.set_node_attributes(G, 'data', { node : discretize(data[node].values, discretizing_bins) for node in G.nodes() })
            nx.set_node_attributes(G, '|Val|', { node : len(set(attr['data'])) for node, attr in G.nodes(data=True) })
        else:
            nx.set_node_attributes(G, 'data', { node : data[node].values for node in G.nodes() })
        return G

    @staticmethod
    def from_graph(G):
        result = DGM()
        result.add_nodes_from(G.nodes())
        result.add_edges_from(G.edges())
        return result


class UGM(nx.Graph):
    """
    Undirected Graphical Model
    """
    def __init__(self):
        super().__init__(self)
        self.potential = permutation_dict()
        self.potential._phook_setitem_ = self._to_factor

    cliques = property(compose(lambda gen: lmap(tuple, gen), nx.find_cliques))

    @property
    def factor_vars(self) -> list:
        return list(self.potential.keys())

    def factors(self):
        result = []
        for key, val in self.potential.items():
            result.append(val.factor())
        return result

    def _to_factor(self, key, obj):
        if isinstance(obj, list) or isinstance(obj, np.ndarray):
            return TableFactor(obj, names=key)
        elif isinstance(obj, dict):
            return DictFactor(obj, names=key)
        elif isinstance(obj, MathFunction):
            return FunctionFactor(obj, names=key)
        return obj

    @property
    def immoralities(self) -> filter:
        """
        Iterate over all immoralities in an unoriented graph.
        :param self: target graph
        :return: iterator over immoralities in form (node, parent1, parent2)
        """
        return filter(lambda v: self.has_edge(v[0], v[1]) and self.has_edge(v[0], v[2]) and (not self.has_edge(v[1], v[2])),
            combinations_with_replacement(self.nodes(), 3))

    plot = pretty_draw

    @staticmethod
    def from_factors(factors):
        """
        Build UGM from a list of factors.
        :param factors: a dict of factors in form  { factor : potential }
        :return: UGM
        """
        ugm = UGM()
        for factor, potential in factors.items():
            ugm.add_nodes_from(factor)
            ugm.add_edges_from(combinations(factor, 2))
            ugm.potential[tuple(factor)] = potential
        return ugm

    def reduce(self, node, val):
        """
        Reduce a node in UGM
        :param node: node to be reduced
        :param val: value assigned to the reduced node
        :return: None
        """
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

    @property
    def factor_graph(self) -> nx.Graph:
        """
        Graph of factors of G: edge exists iff factors intersect.
        :param self: target graph
        :return: graph of factors
        """
        fG = nx.Graph()
        fG.add_nodes_from(self.nodes(), factor=False)
        fG.add_nodes_from(self.factor_vars, factor=True)
        for f in self.factor_vars:
            fG.add_edges_from([(f, var) for var in f])
        return fG

    def fgplot(self):
        """
        Plot graph of factors
        :return: None
        """
        def plot_factor_graph(fG):
            pretty_draw(fG, node_size=lambda node, attr: 1700 if attr['factor'] else 300,
                       node_color=lambda node, attr: '#88FF88' if attr['factor'] else '#AAAAFF')

        return plot_factor_graph(self.factor_graph)

    @property
    def moralize(self) -> nx.Graph:
        """
        Moralize an undirected graph: add edges so that all immoralities are converted to covered
        v-structures. Original graph is not modified, rather, a new graph is created.
        :param G: target graph
        :return: moralized graph
        """
        mG = nx.Graph()
        mG.add_nodes_from(self.nodes())
        mG.add_edges_from(self.edges())
        for x, p1, p2 in self.immoralities:
            mG.add_edge(p1, p2)
        return mG

    @property
    def is_moral(self) -> bool:
        """
        A graph is moral if it has no immoralities.
        :param self: target graph
        :return: is target graph moral
        """
        return len(list(self.immoralities)) == 0

class PDAG(nx.DiGraph):
    def __init__(self):
        super().__init__()

    def has_unoriented_edge(self, x, y):
        return self.has_edge(x, y) and self.has_edge(y, x)

    def has_oriented_edge(self, x, y):
        return self.has_edge(x, y) and not self.has_edge(y, x)

    def has_any_edge(self, x, y):
        return self.has_edge(x, y) or self.has_edge(y, x)

    def orient_edge(self, x, y):
        self.remove_edge(y, x)

    def rule1(self, x, y, z):
        if self.has_oriented_edge(x, y) and self.has_unoriented_edge(y, z) and not self.has_any_edge(x, z):
            self.orient_edge(y, z)
            return True
        return False

    def rule2(self, x, y, z):
        if self.has_oriented_edge(x, y) and self.has_oriented_edge(y, z) and self.has_unoriented_edge(x, z):
            self.orient_edge(x, z)
            return True
        return False

    def rule3(self, x, y1, y2, z):
        if self.has_unoriented_edge(x, y1) and self.has_unoriented_edge(x, y2) and \
                self.has_unoriented_edge(x, z) and not self.has_any_edge(y1, y2) and \
                self.has_oriented_edge(y1, z) and self.has_oriented_edge(y2, z):
            self.orient_edge(x, z)
            return True
        return False

    def _rule3_partial_check1(self, y1, y2, z):
        if self.has_oriented_edge(y1, z) and self.has_oriented_edge(y2, z) and not self.has_any_edge(y1, y2):
            return True
        return False

    def _rule3_partial_check2(self, x, y1, y2, z):
        if self.has_unoriented_edge(x, y1) and self.has_unoriented_edge(x, y2) and self.has_unoriented_edge(x, z):
            self.orient_edge(x, z)
            return True
        return False

    def apply_rule(self):
        for x, y, z in combinations_with_replacement(self.nodes(), 3):
            if self.rule1(x, y, z):
                return True
            if self.rule2(x, y, z):
                return True
            if self._rule3_partial_check1(x, y, z):
                for w in self.nodes():
                    if self._rule3_partial_check2(w, x, y, z):
                        return True
        return False

    def update_directions(self):
        while self.apply_rule():
            pass

    @staticmethod
    def from_dgm(G):
        pdag = PDAG()
        pdag.add_nodes_from(G.nodes())
        for s, t in G.edges():
            pdag.add_edge(s, t)
            pdag.add_edge(t, s)
        pretty_draw(pdag)
        for x, p1, p2 in G.immoralities:
            if pdag.has_edge(x, p1):
                pdag.orient_edge(p1, x)
            if pdag.has_edge(x, p2):
                pdag.orient_edge(p2, x)
        pdag.update_directions()
        return pdag

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

def random_binary_dgm(n=10, p=0.2):
    G = nx.gnr_graph(n, p)
    dgm = DGM()
    dgm.add_nodes_from(G.nodes())
    dgm.add_edges_from(G.edges())
    nx.set_node_attributes(dgm, 'CPD', { node: TableFactor(random_table_factor(dgm.in_degree(node) + 1), list(dgm.predecessors(node)) + [node]) for node in dgm.nodes() })
    return dgm

from itertools import combinations_with_replacement
def clique_graph(G):
    """
    Build a clique graph; nodes are cliques, two nodes are connected iff cliques intersect.
    :param G: target graph
    :return: the clique graph
    """
    cg = nx.Graph()
    for clique in nx.find_cliques(G):
        cg.add_node(tuple(clique))
    for node1, node2 in combinations_with_replacement(list(cg.nodes()), 2):
        if set(node1) & set(node2):
            cg.add_edge(node1, node2)
    return cg

def graph_from_factors(factors):
    """
    Build graph from factors
    :param factors: a list of factors
    :return: graph obtained from factors
    """
    ugm = nx.Graph()
    for factor in factors:
        ugm.add_nodes_from(factor)
        ugm.add_edges_from(combinations(factor, 2))
    return ugm