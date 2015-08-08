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

from .information import mutual_information
from .utility import *
from .information import pairwise_mutual_info, mutual_information, conditional_mutual_information

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
