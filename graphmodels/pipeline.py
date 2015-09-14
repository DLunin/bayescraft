from .representation import DGM
from .factors import Factor
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from .data_preparation import DiscreteDataPreparer

class Pipeline:
    def __init__(self):
        self.reset()

    def reset(self):
        self.graphical_model_type = DGM
        self.structure_learner = None
        self._parameter_estimator = None
        self.inferencer = None
        self.graphical_model = None
        self._data_preparer = None

    @property
    def data_preparer(self):
        return self._data_preparer

    @data_preparer.setter
    def data_preparer(self, val):
        self._data_preparer = val
        if self.parameter_estimator:
            self.parameter_estimator.data_preparer = val

    @property
    def parameter_estimator(self):
        return self._parameter_estimator

    @parameter_estimator.setter
    def parameter_estimator(self, val):
        self._parameter_estimator = val
        if self.data_preparer:
            self._parameter_estimator.data_preparer = self._data_preparer

    def fit_structure(self, data: pd.DataFrame, **options):
        self.graphical_model = self.structure_learner.learn(data, **options)

    def fit(self, data: pd.DataFrame, **options):
        if not self.graphical_model:
            self.fit_structure(data, **options)
        self.parameter_estimator.fit(self.graphical_model, data, **options)
        self.inferencer.fit(self.graphical_model, **options)

    def prob(self, variables: list, observed: pd.DataFrame, **options) -> [Factor]:
        return self.inferencer.prob(variables, observed, **options)

    def predict(self, variables: list, observed: pd.DataFrame, **options) -> pd.DataFrame:
        return self.inferencer.predict(variables, observed, **options)

def misclassification_score(result: pd.DataFrame, truth: pd.DataFrame):
    assert len(result.columns) == len(truth.columns) and len(truth.columns) == 1
    n = result.values.shape[1]
    score = 0.
    for x, y in zip(result.values[:, 0], truth.values[:, 0]):
        if x == y:
            score += 1.
            print('right')
        else:
            print('wrong')
    score /= n
    return score

class CrossValidator:
    def __init__(self, data: pd.DataFrame, score, n_parts: int=5):
        self.data = data
        self.score = score
        self.n_parts = n_parts

    def __call__(self, fit, predict, test_data_preparer, target_columns):
        kf = KFold(len(self.data.values), n_folds=self.n_parts, shuffle=True)
        result = []
        tdata = test_data_preparer(self.data.copy())
        for train_index, test_index in kf:
            train = self.data.iloc[train_index]
            test = tdata.iloc[test_index]
            target = test[target_columns]
            for col in target_columns:
                del test[col]
            fit(train)
            result.append(self.score(predict(target_columns, test), target))
        result = np.array(result)
        return result.mean(), result.std()

