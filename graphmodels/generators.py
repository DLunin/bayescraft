import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from itertools import *
import pytest
from bayescraft.graphmodels.factors import TableCPD

def names_to_str(g):
    result = nx.Graph()
    result.add_nodes_from(map(str, g.node()))
    result.add_edges_from(map(lambda x: (str(x[0]), str(x[1])), g.edges()))
    return result

class AcyclicDiGraphGen:
    @staticmethod
    def diamond(n_var):
        G = nx.DiGraph()
        G.add_nodes_from(range(n_var))
        G.add_edges_from(zip(repeat(0), range(1, n_var-1)))
        G.add_edges_from(zip(range(1, n_var-1), repeat(n_var-1)))
        G.add_edge(0, n_var-1)
        return G

    @staticmethod
    def star(n_var):
        G = nx.DiGraph()
        G.add_nodes_from(range(n_var))
        G.add_edges_from([(i, 0) for i in range(1, n_var)])
        return G

    @staticmethod
    def random_gnr(n_var, p=0.2):
        return nx.gnr_graph(n_var, p)

    @staticmethod
    def random_erdos_renyi(n_var, p=0.2):
        while True:
            G = nx.erdos_renyi_graph(n_var, p, directed=True)
            if not nx.is_directed_acyclic_graph(G):
                continue
            return G


class GraphGen:
    @staticmethod
    def diamond(n_var):
        G = nx.Graph()
        G.add_nodes_from(range(n_var))
        G.add_edges_from(zip(repeat(0), range(1, n_var-1)))
        G.add_edges_from(zip(range(1, n_var-1), repeat(n_var-1)))
        G.add_edge(0, n_var-1)
        return G

    @staticmethod
    def star(n_var):
        return nx.star_graph(n_var)

    @staticmethod
    def random_erdos_renyi(n_var, p=0.2):
        return nx.erdos_renyi_graph(n_var, p)


class DiscreteModelGenDGM:
    @staticmethod
    def dirichlet(G, alpha=1):
        cpd = {}
        for node in nx.topological_sort(G):
            m = G.in_degree(node) + 1
            dim = tuple([2] * m)
            table = stats.dirichlet(alpha=tuple([alpha] * (2 ** m))).rvs()[0]
            table = table.reshape(dim)
            names = [node] + list(G.predecessors(node))
            cpd[node] = TableCPD(table, names)
        return cpd

class ContinuousModelGenDGM:
    pass