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
from itertools import repeat
import heapq
import pandas as pd
from collections import deque


from .information import mutual_information
from .utility import *
from .information import pairwise_mutual_info, mutual_information, conditional_mutual_information, discrete_mutual_information
from .representation import DGM

def build_pmap_skeleton(names, ci_test, d=None):
    """
    Build the skeleton of the P-map using the witness sets.
    :param names: variable names
    :param ci_test: a conditional independence oracle of the form (name1, name2, witness -> bool)
    :param d: maximum number of parents in graph
    :return: P-map skeleton graph, set of witnesses
    """
    if d is None:
        d = len(names) - 1
    G = nx.Graph()
    G.add_nodes_from(names)
    G.add_edges_from(combinations(names, 2))

    witnesses = {}
    for x, y in combinations(names, 2):
        print(x, y)
        x_neigh = list(G.neighbors(x))
        y_neigh = list(G.neighbors(y))
        for witness in chain(*([combinations(x_neigh, i) for i in range(1, 1 + min(len(x_neigh), d))] + \
                             [combinations(y_neigh, i) for i in range(1, 1 + min(len(y_neigh), d))])):
            if ci_test(x, y, witness):
                witnesses[x, y] = witness
                G.remove_edge(x, y)
                break
    return G, witnesses

def info_ci_test(x, y, witness, treshold=None, data=None):
    """
    Conditional independence test based on conditional mutual information
    :param x: first variable name
    :param y: second variable name
    :param witness: witness set
    :param treshold: treshold for mutual information
    :param data: the dataset
    :return: are the variables independent
    """
    return conditional_mutual_information(data, x, witness, y) < treshold

def chow_liu(X):
    """
    Chow-Liu structure learning algorithm.
    :param X: dataset
    :return: the learned graph (tree)
    """
    n_objects = X.shape[0]
    n_vars = X.shape[1]
    g = nx.complete_graph(n_vars)
    for i, j in g.edges():
        g.edge[i][j]['mutual information'] = mutual_info_score(X[:, i], X[:, j])
    g = maximum_spanning_tree(g, weight='mutual information')
    return g

class BIC_score:
    """
    BIC (Bayes Information Criterion) score of DGM graph. All terms which don't depend
    on the network structure are thrown out.
    """
    def __init__(self, G, data, value_set_size_attr='|Val|'):
        self.G = G
        self.data = data
        self.cache = { }
        self.m = self.data.shape[0]
        self.value_set_size_attr = value_set_size_attr
        self.value = sum([self.famscore(node) for node in self.G.nodes()])

    def famscore(self, node):
        """
        Family score of a single node. BIC score is decomposable, so total score is the sum
        of family scores of all nodes.
        """

        def dimension(node):
            """
            Number of free parameters in node CPD.
            """
            result = self.G.node[node][self.value_set_size_attr]
            for parent in self.G.predecessors(node):
                result *= self.G.node[parent][self.value_set_size_attr]
            return result - 1

        if node in self.cache:
            return self.cache[node]
        dim = dimension(node)
        result = - dim * log(self.m) / 2
        if self.G.in_degree(node) > 0:
            result += self.m * discrete_mutual_information(self.data[[node]].values, self.data[list(self.G.predecessors(node))].values)
        self.cache[node] = result
        return result

    def invalidate(self, node):
        """
        Notifies score calculator that the family of the node has changed, in order to update its
        score.
        """
        if node in self.cache:
            self.value -= self.cache[node]
            self.cache.pop(node)
        self.value += self.famscore(node)

    def reset(self):
        """
        Recalculate all the scores.
        :return: None
        """
        for node in self.G.nodes():
            self.invalidate(node)
        return None

class StrucutreSearchEdgeOp:
    def __init__(self, G, scoring):
        self.G = G
        self.scoring = scoring

    def __call__(self):
        G = self.G

        for s, t in combinations(self.G.nodes(), 2):
            yield (self, (s, t))

    def score(self, s, t):
        prev = self.scoring.value
        if self.apply(s, t):
            result = self.scoring.value - prev
            self.cancel(s, t)
            return result
        return 0.

    def apply(self, s, t):
        return True

    def cancel(self, s, t):
        pass

class EdgeAddOp(StrucutreSearchEdgeOp):
    def __init__(self, G, scoring):
        super().__init__(G, scoring)

    def _apply(self, s, t):
        self.G.add_edge(s, t)

    def apply(self, s, t):
        if self.G.has_edge(s, t):
            return False
        self._apply(s, t)
        if not nx.is_directed_acyclic_graph(self.G):
            self._cancel(s, t)
            return False
        self.scoring.invalidate(t)
        return True

    def _cancel(self, s, t):
        self.G.remove_edge(s, t)

    def cancel(self, s, t):
        self._cancel(s, t)
        self.scoring.invalidate(t)

    def is_affected(self, nodes, s, t):
        return t in nodes

    def affects(self, s, t):
        return [t]

class EdgeRemoveOp(StrucutreSearchEdgeOp):
    def __init__(self, G, scoring):
        super().__init__(G, scoring)

    def apply(self, s, t):
        if not self.G.has_edge(s, t):
            return False
        self._apply(s, t)
        self.scoring.invalidate(t)
        return True

    def _apply(self, s, t):
        self.G.remove_edge(s, t)

    def cancel(self, s, t):
        self._cancel(s, t)
        self.scoring.invalidate(t)

    def _cancel(self, s, t):
        self.G.add_edge(s, t)

    def is_affected(self, nodes, s, t):
        return t in nodes

    def affects(self, s, t):
        return [t]


class EdgeReverseOp(StrucutreSearchEdgeOp):
    def __init__(self, G, scoring):
        super().__init__(G, scoring)

    def apply(self, s, t):
        if not self.G.has_edge(s, t):
            return False
        self._apply(s, t)
        if nx.is_directed_acyclic_graph(self.G):
            self.scoring.invalidate(s)
            self.scoring.invalidate(t)
            return True
        else:
            self._cancel(s, t)
            return False

    def _apply(self, s, t):
        self.G.remove_edge(s, t)
        self.G.add_edge(t, s)

    def cancel(self, s, t):
        self.apply(t, s)

    def _cancel(self, s, t):
        return self._apply(t, s)

    def is_affected(self, nodes, s, t):
        return t in nodes or s in nodes

    def affects(self, s, t):
        return [s, t]

def bagging(data):
    headers = list(data.columns.values)
    data = data.values
    result = data[np.random.randint(data.shape[0], size=data.shape[0])]
    return pd.DataFrame(data=result, columns=headers)

class StructureSearch:
    """
    Class for performing local structure search.
    """
    def __init__(self, data, scoring, operations):
        self.data = data
        self.scoring = scoring
        self.operations = operations

    def __call__(self, G, n_iterations=1000, do_bagging=True, taboo_len=0):
        """
        Does the structure search.
        :param G: target graph
        :param n_iterations: maximum number of iterations
        :return: result graph
        """
        data = self.data
        score = self.scoring(G, data)

        bagging_iter = -50

        operations = list(map(lambda Op: Op(G, score), self.operations))
        opdata = sum([list(op()) for op in operations], [])
        operations_heap = lmap(lambda i: (-opdata[i][0].score(*opdata[i][1]), i), range(len(opdata)))
        taboo = deque([-1 for i in range(taboo_len)])

        for n_iter in range(n_iterations):
            print('Iteration', n_iter)

            affects = []
            best_score = -np.inf

            operations_heap.sort()
            for i in range(len(operations_heap)):
                best_score, idx = operations_heap[i]
                op, args = opdata[idx]
                best_score = -best_score
                prev_score = score.value
                if idx not in taboo and op.apply(*args):
                    taboo.append(idx)
                    taboo.popleft()
                    if abs(score.value - prev_score - best_score) > 0.00001:
                        op.cancel(*args)
                        continue
                    affects = op.affects(*args)
                    break

            if best_score <= 0:
                op.cancel(*args)
                if do_bagging and n_iter - bagging_iter > 10:
                    print('bagging data')
                    score.data = bagging(score.data)
                    score.reset()
                    best_score = score.value
                    bagging_iter = n_iter
                else:
                    break
            else:
                for i in range(len(operations_heap)):
                    neg_score, idx = operations_heap[i]
                    op, (s, t) = opdata[idx]
                    if op.is_affected(affects, s, t):
                        operations_heap[i] = -op.score(s, t), idx
        return G