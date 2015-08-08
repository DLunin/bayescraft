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

from graphmodels.flow_analysis import *

def dgm_sum_model(G, distr, n_samples):
    data = { }
    n_var = nx.number_of_nodes(G)
    for node in nx.topological_sort(G):
        if G.in_degree(node) > 0:
            predecessors_data = np.hstack([data[v] for v in G.predecessors(node)])
            data[node] = distr(n_samples, 1) + np.sum(predecessors_data, axis=1).reshape((n_samples, 1))
        else:
            data[node] = distr(n_samples, 1)
    return data

def diamond_testcase(n_var, n_samples, distr=np.random.randn):
    G = nx.DiGraph()
    G.add_nodes_from(range(n_var+2))
    G.add_edges_from(zip(repeat(0), range(1, n_var+1)))
    G.add_edges_from(zip(range(1, n_var+1), repeat(n_var+1)))
    G.add_edge(0, n_var + 1)
    return G, dgm_sum_model(G, distr, n_samples)

def stargraph_ugm_testcase(n_var, n_samples):
    G = nx.star_graph(n_var)
    mean = 10 * np.random.rand(n_samples, 1) - 5
    samples = mean + np.random.randn(n_samples, n_var)
    return G, np.hstack([mean, samples])

def stargraph_dgm_testcase(n_var, n_samples):
    G = nx.DiGraph()
    G.add_nodes_from(range(n_var + 1))
    G.add_edges_from([(i, 0) for i in range(1, n_var+1)])
    samples = np.random.randn(n_samples, n_var)
    center = np.sum(samples, axis=1).reshape((n_samples, 1))
    return G, np.hstack([center, samples])

def test_flowdiff_small():
    G = nx.DiGraph()
    G.add_nodes_from(range(3))
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    n_samples = 100
    data = dgm_sum_model(G, np.random.randn, n_samples)
    data = np.hstack([data[i] for i in range(3)])

    G.remove_edge(0, 2)
    assert flowdiff(flowgraph(G, data), data, 0, 2) > -0.1
    G.add_edge(0, 2)
    assert flowdiff(flowgraph(G, data), data, 0, 2) < 0.1
    G.remove_edge(1, 2)
    assert flowdiff(flowgraph(G, data), data, 1, 2) > -0.1
    G.add_edge(1, 2)
    assert flowdiff(flowgraph(G, data), data, 1, 2) < 0.1

def test_gomory_hu():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, info=10)
    G.add_edge(0, 2, info=5)
    G.add_edge(1, 2, info=4)
    T = gomory_hu_tree(G, capacity='info')
    assert list(T.edges(data=True)) == [(0, 1, {'info': 14.0}), (0, 2, {'info': 9.0})]

def test_edge_candidates():
    def run_gnr(n_var, n_samples, p_edge):
        G = nx.gnr_graph(n_var, p_edge)
        data = dgm_sum_model(G, np.random.randn, n_samples)
        data = np.hstack([data[i] for i in range(n_var)])
        edge = random.choice(list(G.edges()))
        G.remove_edge(*edge)
        flowdiffs = edge_candidates(G, data)
        flowedges = list(sum([[(v1, v2), (v2, v1)] for score, v1, v2 in flowdiffs[:-10:-1]], []))
        return edge in flowedges

    for i in range(10):
        assert run_gnr(n_var=10, n_samples=100, p_edge=0.4)