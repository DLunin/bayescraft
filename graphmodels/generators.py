import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from itertools import *
import pytest
from bayescraft.graphmodels.factors import (TableCPD, MultivariateGaussianDistribution,
                                            ParametricFunctionCPD, LinearGaussianDistribution)
from bayescraft.graphmodels import DGM
import bayescraft.stats as bstats

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
    @staticmethod
    def gaussian(G):
        cpd = {}
        for node in nx.topological_sort(G):
            m = G.in_degree(node) + 1
            cov = np.random.rand(m, m)
            cov = np.dot(cov, cov.T)
            d = MultivariateGaussianDistribution(np.zeros(m), cov)
            cpd[node] = ParametricFunctionCPD(d, [node] + list(G.predecessors(node)))
        return cpd

    @staticmethod
    def linear_gaussian(G, a_0=1, b_0=1):
        cpd = {}
        for node in nx.topological_sort(G):
            m = G.in_degree(node) + 1
            nig = bstats.normal_inverse_gamma(w_0=np.zeros(m), V_0=np.eye(m), a_0=a_0, b_0=b_0)
            sample = nig.rvs()
            variance = sample[-1]
            w = sample[1:-1]
            w0 = sample[0]
            cpd[node] = ParametricFunctionCPD(LinearGaussianDistribution(w0, w, variance),
                                              [node], list(G.predecessors(node)))
        return cpd

def dag_pack():
    for n_var in [5, 10, 20, 30]:
        yield AcyclicDiGraphGen.diamond(n_var)
    for n_var in [5, 10, 20, 30]:
        yield AcyclicDiGraphGen.star(n_var)
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]:
        for n_var in [5, 10, 20, 30]:
            yield AcyclicDiGraphGen.random_gnr(n_var, p)
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]:
        for n_var in [5, 10, 20, 30]:
            yield AcyclicDiGraphGen.random_erdos_renyi(n_var, p)

def dgm_pack():
    for dag in dag_pack():
        dgm = DGM.from_graph(dag)
        dgm.cpd = DiscreteModelGenDGM.dirichlet(dag.copy())
        #yield dgm
        dgm = DGM.from_graph(dag)
        dgm.cpd = ContinuousModelGenDGM.linear_gaussian(dgm)
        yield dgm

