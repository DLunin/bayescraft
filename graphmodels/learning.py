import pandas as pd
import numpy as np
from .representation import DGM
from .factors import TableCPD
from .data_preparation import DiscreteDataPreparer

class ParameterEstimator:
    def fit(self, graphical_model, data: pd.DataFrame, **options):
        raise NotImplementedError()

class MleParameterEstimator:
    def __init__(self, data_preparer=None):
        self.data_preparer = data_preparer

    def fit(self, graphical_model, data: pd.DataFrame, **options):
        data = self.data_preparer(data.copy())
        assert(isinstance(graphical_model, DGM))
        graphical_model.model = { x : TableCPD for x in data.columns.values }
        cardinality = options['pe_cardinality'] if 'pe_cardinality' in options else { x : max(data[x]) + 1 for x in data.columns.values }
        graphical_model.cpd = graphical_model.mle(data, cardinality)
