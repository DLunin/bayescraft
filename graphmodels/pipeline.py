from .representation import DGM
from .factors import Factor
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold


class Pipeline:
    def __init__(self):
        self.reset()

    def reset(self):
        self.graphical_model_type = DGM
        self.structure_learner = None
        self.parameter_estimator = None
        self.inferencer = None
        self.graphical_model = None

    def fit_structure(self, data: pd.DataFrame, **options):
        self.graphical_model = self.structure_learner.learn(data, **options)

    def fit(self, data: pd.DataFrame, **options):
        if not self.graphical_model:
            self.fit_structure(data, **options)
        self.parameter_estimator.fit(self.graphical_model, data, **options)
        self.inferencer.fit(self.graphical_model, **options)

    def prob(self, variables: list, observed: pd.DataFrame) -> [Factor]:
        self.inferencer.prob(variables, observed)

    def predict(self, variables: list, observed: pd.DataFrame) -> pd.DataFrame:
        self.inferencer.predict(variables, observed)


class CrossValidator:
    def __init__(self, data: pd.DataFrame, score, n_parts: int=5):
        self.data = data
        self.score = score
        self.n_parts = n_parts

    def __call__(self, fit, predict, target_columns):
        kf = KFold(len(self.data.values), n_folds=self.n_parts, shuffle=True)
        result = []
        for train_index, test_index in kf:
            train = self.data.iloc[train_index]
            test = self.data.iloc[test_index]
            target = test[target_columns]
            for col in target_columns:
                del test[col]
            fit(train)
            result.append(self.score(predict(test, target), target))
        result = np.array(result)
        return result.mean(), result.std()
