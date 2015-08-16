import networkx as nx
from functools import wraps

class Graph:
    def __init__(self):
        self._impl = nx.Graph()

    def add_node(self, node):
        return self._impl.add_node(node)

    def remove_node(self, node):
        return self._impl.remove_node(node)

    def add_edge(self, s, t):
        return self._impl.add_edge(s, t)

    def remove_edge(self, s, t):
        return self._impl.remove_edge(s, t)
