import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import product
import random
import math
from functools import wraps

def compose(f, g):
    """
    :param f: Second function to apply
    :param g: First function to apply
    :return: A composition of functions
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))

lmap = compose(list, map)

def sigmoid(x):
    """
    Logistic sigmoid function
    :param x: argument
    :return: function value
    """
    return 1 / (1 + math.exp(-x))


def extract_node_attribute(graph, name, default=None):
    """
    Extract attributes of a networx graph nodes to a dict.
    :param graph: target graph
    :param name: name of the attribute
    :param default: default value (used if node doesn't have the specified attribute)
    :return: a dict of attributes in form of { node_name : attribute }
    """
    return { i : d.get(name, default) for i, d in graph.nodes(data=True) }


def extract_edge_attribute(graph, name, default=None):
    """
    Extract attributes of a networx graph edges to a dict.
    :param graph: target graph
    :param name: name of the attribute
    :param default: default value (used if edge doesn't have the specified attribute)
    :return: a dict of attributes in form of { (from, to) : attribute }
    """
    return { (i, j) : d.get(name, default) for i, j, d in graph.edges(data=True) }


def pretty_draw(graph, node_color=lambda node, attr: '#DDDDDD',
                edge_color=lambda node1, node2, attr: '#000000', node_size=lambda node, attr: 300):
    """
    Draws a graph. You can specify colors of nodes, colors of edges and size of nodes via lambda
    functions.
    :param graph: target graph
    :param node_color: lambda function mapping node name and its attributes to the desired color
    :param edge_color: lambda function mapping edge and its attributes to the desired color
    :param node_size: lambda function mapping node name and its attributes to the desired size
    :return: None
    """
    fig = plt.figure(figsize=(17, 6))
    plt.axis('off')
    if type(node_color) is str:
        node_colors = extract_node_attribute(graph, 'color', default='#DDDDDD')
        node_colors = list(map(node_colors.__getitem__, graph.nodes()))
    else:
        node_colors = list(map(lambda args: node_color(*args), graph.nodes(data=True)))

    if type(edge_color) is str:
        edge_colors = extract_edge_attribute(graph, 'color', default='#000000')
        edge_colors = list(map(edge_colors.__getitem__, graph.edges()))
    else:
        edge_colors = list(map(lambda args: edge_color(*args), graph.edges(data=True)))

    if type(node_size) is str:
        node_sizes = extract_node_attribute(graph, 'size', default='300')
        node_sizes = list(map(node_sizes.__getitem__, graph.nodes()))
    else:
        node_sizes = list(map(lambda args: node_size(*args), graph.nodes(data=True)))

    nx.draw_networkx(graph,
                     with_labels=True,
                     pos=nx.spring_layout(graph),
                     node_color=node_colors,
                     edge_color=edge_colors,
                     node_size=node_sizes
                     )
    return None

def maximum_spanning_tree(graph, weight='weight'):
    """
    Find a maximum spanning tree of a graph
    :param graph: target graph
    :param weight: edge attribute which will be used as edge weight
    :return: maximum spanning tree graph (networkx.Graph)
    """
    for i, j in graph.edges():
        graph.edge[i][j][weight] = -graph.edge[i][j][weight]
    result = nx.minimum_spanning_tree(graph, weight='weight')
    for i, j in graph.edges():
        graph.edge[i][j][weight] = -graph.edge[i][j][weight]
    return result

def plot_distr_2d(distr, domain=(-25, 25)):
    """
    Smart 1d probability distribution plotter. Finds out the interval where the most of probability
    mass lies, and plots distribution on it (so you don't need to specify x-axis interval).
    :param distr: distribution to plot in (vectorized) form of numpy.array<float> -> numpy.array<float>
    :param domain: a superset of plotting interval (to narrow search)
    :return: None
    """
    def binary_search_quantiles(quantile, begin, end, prec):
        while end - begin > prec:
            sep = (begin + end) / 2.0
            #print(sep, sp.integrate.quad(distr, -np.inf, sep)[0])
            if sp.integrate.quad(distr, -np.inf, sep)[0] < quantile:
                begin = sep
            else:
                end = sep
        return (begin + end) / 2.0

    alpha = 0.001
    begin = binary_search_quantiles(alpha, domain[0], domain[1], 0.1)
    end = binary_search_quantiles(1 - alpha, domain[0], domain[1], 0.1)
    if abs(end - begin) < 1e-10:
        begin, end = domain
    x = np.arange(begin, end, (end - begin) / 1000)
    plt.plot(x, distr(x))
    return None

def plot_distr_3d(distr):
    """
    Plot 2d probability distribution.
    :param distr: the probability distribution to plot in form of [float, float] -> float
    :return: None
    """
    print(distr([0, 0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.squeeze(np.array([[distr([X[i][j], Y[i][j]]) for j in range(X.shape[1])] for i in range(X.shape[0])]))
    ax.plot_surface(X, Y, Z, color='#DDDDDD')
    return None

def plot_distr(distr, dim=1, domain=(-25, 25)):
    """
    Smart distribution plotting (whether 1d or 2d).
    :param distr: the distribution to plot
    :param dim: dimensionality (if known)
    :param domain: domain for 1d version
    :return: None
    """
    if dim == 1:
        try:
            plot_distr_2d(distr, domain=domain)
        except:
            plot_distr_3d(distr)
    else:
        plot_distr_3d(distr)
    return None

def flip_edge(graph, edge):
    """
    Flips an edge in a networkx graph.
    :param graph: a target graph
    :param edge: edge to flip
    :return: None
    """
    if graph.has_edge(*edge):
        graph.remove_edge(*edge)
    else:
        graph.add_edge(*edge)
    return None

def spoil_graph(graph, p):
    """
    'Spoils' a graph: flips every edge with probability p. Doesn't change the original graph.
    :param graph: target graph
    :param p: flip probability
    :return: spoiled graph
    """
    graph = graph.copy()
    for i in range(len(graph.nodes())):
        for j in range(i):
            if random.random() < p:
                flip_edge(graph, (i, j))
    return graph

def reverse_edge(G, edge, copy=False):
    """
    Reverse edge in graph.
    :param G: target graph
    :param edge: target edge
    :param copy: if True, copy graph before changing it
    :return: graph with reversed edge
    """
    if copy:
        G = G.copy()
    x, y = edge
    G.remove_edge(x, y)
    G.add_edge(y, x)
    return G

def are_equal_graphs(G1, G2):
    """
    Check graph equality (equal node names, and equal edges between them).
    :param G1: first graph
    :param G2: second graph
    :return: are they equal
    """
    if set(G1.nodes()) != set(G2.nodes()):
        return False
    return all(map(lambda x: G1.has_edge(*x), G2.edges())) and all(map(lambda x: G2.has_edge(*x), G1.edges()))

def is_subgraph(G1, G2):
    """
    Is G1 a subgraph of G2?
    :param G1: supposed subgraph
    :param G2: graph
    :return: is G1 subgraph of G2
    """
    return set(G1.edges()).issubset(set(G2.edges()))

def descendants(G, x):
    """
    Set of all descendants of node in a graph, not including itself.
    :param G: target graph
    :param x: target node
    :return: set of descendants
    """
    return set(nx.dfs_preorder_nodes(G, x)) - {x}

def ancestors(G, x, G_reversed=None):
    """
    Set of all ancestors of node in a graph, not including itself.
    :param G: target graph
    :param x: target node
    :param G_reversed: you can supply graph with reversed edges for speedup
    :return: set of ancestors
    """
    if G_reversed is None:
        G_reversed = G.reverse()
    return descendants(G_reversed, x)


def reprsort(li):
    """
    sometimes, we need a way to get an unique ordering of any Python objects
    so here it is!
    (not quite "any" Python objects, but let's hope we'll never deal with that)
    """
    extli = list(zip(map(repr, li), range(len(li))))
    extli.sort()
    return [li[i[1]] for i in extli]


class ListTable(list): # from http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/
    """
    Overridden list class which takes a 2-dimensional list of
    the form [[1,2,3],[4,5,6]], and renders an HTML Table in
    IPython Notebook.
    """

    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")

            for col in row:
                html.append("<td>{0}</td>".format(col))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def pretty_print_distr_table(table, names):
    """
    Get a ListTable of the distribution specified by `table`, so that it can be
    prettily rendered in ipython notebook
    :param table: table of the distribution
    :param names: names assigned to variables in table
    :return: ListTable
    """
    table = np.array(table)
    t = ListTable()
    t.append(names + ['P'])
    for v in product(*lmap(compose(list, range), table.shape)):
        t.append(list(v) + [str(table[v])[:5]])
    return t

def pretty_print_distr_dict(d, names):
    """
    Get a ListTable of the distribution specified by dict `d`, so that it can be
    prettily rendered in ipython notebook.
    :param d: dict of the distribution
    :param names: names assigned to variables in dict
    :return: ListTable
    """
    t = ListTable()
    t.append(names + ['P'])

    items = list(d.items())
    try:
        items.sort()
    except TypeError:
        items = reprsort(items)

    for v, p in items:
        t.append(list(v) + [str(p)[:5]])
    return t

class permutation_dict(dict):
    """
    A modification of dict.

    Tuple keys are considered equal, if the first can be obtained by permuting the second.
    For example (1, 3, 2, 0) == (0, 1, 2, 3)

    Also, hooks for __getitem__ and __setitem__ are provided.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phook_setitem_ = lambda key, val: val
        self._phook_getitem_ = lambda key, val: val

    def __setitem__(self, arg, val):
        if isinstance(arg, tuple):
            arg = reprsort(list(arg))
            arg = tuple(arg)
        else:
            arg = tuple([arg])
        val = self._phook_setitem_(arg, val)
        return super().__setitem__(arg, val)

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            arg = reprsort(list(arg))
            arg = tuple(arg)
        else:
            arg = tuple([arg])
        return self._phook_getitem_(arg, super().__getitem__(arg))

def stabilize(alpha):
    """
    Decorator which tries to reduce variance of a random function by
    averaging it across multiple calls. Function must return a float.
    :param alpha: required precision
    :return: stabilized function
    """
    def stabilize_decorator(f):
        @wraps(f)
        def new_f(*args, **kwargs):
            x = 0.0
            current = f(*args, **kwargs)
            n = 1
            while abs(x - current) > alpha or n == 1:
                prev = current
                x = f(*args, **kwargs)
                current = (n / (n + 1)) * current + (x / (n + 1))
                n += 1
            return current
        return new_f
    return stabilize_decorator

def relmatrix(f, val1, val2):
    """
    A table (2d numpy array) obtained by applying function `f` to different combinations of
    values from `val1` and `val2`
    :param f: applied function
    :param val1: row values
    :param val2: col values
    :return: numpy array -- the table
    """
    res = [[''] + list(val2)]
    for v1 in val1:
        li = [v1]
        for v2 in val2:
            li.append(f(v1, v2))
        res.append(li)
    return res

def infotable(data):
    """
    Table of pairwise mutual informations between variables in the dataset.
    :param data: the dataset
    :return: the resulting table
    """
    n_var = data.shape[1]
    return [[mutual_information(data[:, i1:i1+1], data[:, i2:i2+1]) for i2 in range(n_var)] for i1 in range(n_var)]

def infomatrix(data):
    """
    Table of pairwise mutual informations between variables in the dataset in the form of ListTable
    :param data: the dataset
    :return: the resulting table as ListTable
    """
    n_var = data.shape[1]
    return ListTable(relmatrix(lambda i1, i2: mutual_information(data[:, i1:i1+1], data[:, i2:i2+1]), range(n_var), range(n_var)))
