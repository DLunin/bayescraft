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
from .utility import maximum_spanning_tree, ListTable, lmap
from .information import pairwise_mutual_info, mutual_information

def build_pmap_skeleton(names, ci_test, d=None):
    if d is None:
        d = len(names) - 1
    G = nx.Graph()
    G.add_nodes_from(names)
    G.add_edges_from(combinations(names, 2))

    witnesses = {}
    for x, y in combinations(names, 2):
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
    return mutual_information(data[x], data[y], data[witness])

def chow_liu(X):
    n_objects = X.shape[0]
    n_vars = X.shape[1]
    g = nx.complete_graph(n_vars)
    for i, j in g.edges():
        g.edge[i][j]['mutual information'] = mutual_info_score(X[:, i], X[:, j])
    g = maximum_spanning_tree(g, weight='mutual information')
    return g

def relmatrix(f, val1, val2):
    res = [[''] + list(val2)]
    for v1 in val1:
        li = [v1]
        for v2 in val2:
            li.append(f(v1, v2))
        res.append(li)
    return res

def infotable(data):
    n_var = data.shape[1]
    return [[mutual_information(data[:, i1:i1+1], data[:, i2:i2+1]) for i2 in range(n_var)] for i1 in range(n_var)]

def infomatrix(data):
    n_var = data.shape[1]
    return ListTable(relmatrix(lambda i1, i2: mutual_information(data[:, i1:i1+1], data[:, i2:i2+1]), range(n_var), range(n_var)))

from collections import deque
# поиск кратчайшего пути с помощью dfs
def bfs_path(G, u, v, blocked = lambda u, v: False):
    visited = set([u])
    prev = { u : None }
    queue = deque()
    queue.append(u)
    while True:
        if not queue:
            raise nx.NetworkXNoPath()
        current = queue.popleft()
        if current == v:
            break
        for succ in G.successors(current):
            if succ in visited:
                continue
            if blocked(current, succ):
                continue
            queue.append(succ)
            prev[succ] = current
            visited.add(succ)
    result = []
    current = v
    while current:
        result.append(current)
        current = prev[current]
    result.reverse()
    return result

# Модифицированный Форд-Фалкерсон
def info_flow_func(G, s, t, capacity='capacity', value_only=False):
    eps = 1e-9
    RG = nx.DiGraph()
    RG.add_nodes_from(G.nodes())
    for v1, v2 in G.edges():
        if RG.has_edge(v1, v2):
            RG[v1][v2]['capacity'] = G[v1][v2][capacity]
        RG.add_edge(v1, v2, capacity=G[v1][v2][capacity], flow=0)
        RG.add_edge(v2, v1, capacity=0, flow=0)

    flow_value = 0
    while True:
        try:
            path_nodes = bfs_path(RG, s, t, blocked = lambda u, v: RG[u][v]['capacity'] - RG[u][v]['flow'] <= eps)
        except nx.NetworkXNoPath:
            break

        # Get the list of edges in the shortest path.
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

        # Find the minimum capacity of an edge in the path.
        path_capacity = np.inf
        for u, v in path_edges:
            path_capacity = min(RG[u][v]['capacity'] - RG[u][v]['flow'], path_capacity)

        flow_value += path_capacity

        # Augment the flow along the path.
        for u, v in path_edges:
            RG[u][v]['flow'] += path_capacity
            RG[v][u]['flow'] -= path_capacity
            if v[1] == 'bottom':
                if u[1] == 'top':
                    w = (u[0], 'bottom')
                else:
                    w = (u[0], 'top')
                RG[w][v]['flow'] += path_capacity
                RG[v][w]['flow'] -= path_capacity

    RG.graph['flow_value'] = flow_value
    return flow_value


def flowgraph(G, data, directed=True):
    if directed:
        fG = nx.DiGraph()
    else:
        fG = nx.Graph()
    fG.add_nodes_from(zip(G.nodes(), repeat('bottom')))
    fG.add_nodes_from(zip(G.nodes(), repeat('top')))
    for v1, v2 in G.edges():
        info = pairwise_mutual_info(data, v1, v2)
        fG.add_edge((v2, 'top'), (v1, 'top'), info=info)
        fG.add_edge((v1, 'top'), (v2, 'bottom'), info=info)
        fG.add_edge((v1, 'bottom'), (v2, 'bottom'), info=info)
    G.graph['flowgraph'] = fG
    return fG

def infoflow(fG, s, t):
    flow_val1 = info_flow_func(fG, (s, 'top'), (t, 'bottom'), capacity='info')
    flow_val2 = info_flow_func(fG, (s, 'top'), (t, 'top'), capacity='info')
    if flow_val1 > flow_val2:
        return flow_val1
    else:
        return flow_val2

#@stabilize(0.01)
def flowdiff(fG, data, s, t, debug=False):
    flow_val = infoflow(fG, s, t)
    direct_info = pairwise_mutual_info(data, s, t)
    return direct_info - flow_val

def gomory_hu_tree(G, capacity='capacity'):
    import igraph as ig
    g_nodes = list(G.nodes())
    V = len(g_nodes)
    names_to_nodes = dict(zip(g_nodes, range(V)))
    nodes_to_names = dict(enumerate(g_nodes))
    iG = ig.Graph()
    iG.add_vertices(V)
    for v1, v2 in G.edges():
        iG.add_edge(names_to_nodes[v1], names_to_nodes[v2], capacity=G[v1][v2][capacity])
    iT = iG.gomory_hu_tree(capacity='capacity')
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    for edge in iT.es:
        T.add_edge(nodes_to_names[edge.source], nodes_to_names[edge.target], **{ capacity : edge['flow'] })
    return T

class MaxFlows:
    def __init__(self, G, capacity='capacity'):
        self.capacity = capacity
        self.ghtree = gomory_hu_tree(G, capacity=capacity)
        self.G = G

    def __getitem__(self, edge):
        path = nx.shortest_path(self.ghtree, edge[0], edge[1])
        result = min(map(lambda v: self.ghtree[v[0]][v[1]][self.capacity], zip(path, path[1:])))
        return result

    def minflow_edge(self, edges=None):
        if edges is None:
            edges = combinations(self.ghtree.nodes(), 2)
        combs = list(edges)
        flows = lmap(self.__getitem__, combs)
        return combs[flows.index(min(flows))]

    def flowdiff(self, data, s, t):
        return pairwise_mutual_info(data, s, t) - max(self[(s, 'top'), (t, 'bottom')],
                                                      self[(s, 'bottom'), (t, 'top')])

def edge_candidates_undirected(G, data):
    flowdiffs = []
    fG = flowgraph(G, data, directed=False)
    mf = MaxFlows(fG, capacity='info')
    for v1, v2 in combinations(list(G.nodes()), 2):
        flowdiffs.append((mf.flowdiff(data, v1, v2), v1, v2))
        #flowdiffs.append((flowdiff(fG, data, v1, v2), v1, v2))
    flowdiffs.sort()
    return flowdiffs

def edge_candidates(G, data):
    flowdiffs = []
    fG = flowgraph(G, data, directed=True)
    mf = MaxFlows(fG, capacity='info')
    for v1, v2 in combinations(list(G.nodes()), 2):
        #flowdiffs.append((mf.flowdiff(data, v1, v2), v1, v2))
        flowdiffs.append((flowdiff(fG, data, v1, v2), v1, v2))
    flowdiffs.sort()
    return flowdiffs