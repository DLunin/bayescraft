import pandas as pd
import numpy as np
from .representation import DGM
from .factors import TableCPD

class ParameterEstimator:
    def fit(self, graphical_model, data: pd.DataFrame, **options):
        raise NotImplementedError()

class MleParameterEstimator:
    def __init__(self):
        pass

    def fit(self, graphical_model, data: pd.DataFrame, **options):
        assert(isinstance(graphical_model, DGM))
        graphical_model.model = { x : TableCPD for x in data.columns.values }
        cardinality = options['pe_cardinality'] if 'pe_cardinality' in options else { x : max(data[x]) + 1 for x in data.columns.values }
        graphical_model.cpd = graphical_model.mle(data, cardinality)
