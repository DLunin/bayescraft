import networkx as nx
import numpy as np
from numpy import log, exp
import pandas as pd
import scipy as sp
import scipy.stats as stats
import bayescraft.stats as bstats
from itertools import *

from bayescraft.graphmodels.representation import *
from bayescraft.graphmodels.utility import *
import pytest

def test_d_separation(d_separation_dgm):
    dgm = DGM()
    dgm.add_nodes_from('WXYZ')
    dgm.add_edges_from([('W', 'Y'),
                          ('Z', 'Y'),
                          ('Z', 'X'),
                          ('Y', 'X')])
    assert dgm.reachable('X', ['Y'], debug=False) == {'Z', 'W'}

