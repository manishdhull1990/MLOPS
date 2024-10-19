from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer

import dill
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

class CustomNumericalImputer(BaseEstimator,TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
        self.cols = cols
        self.strategy = strategy

    def transform(self, X):
        X = X.copy()
        impute = SimpleImputer(strategy=self.strategy)
        if self.cols == None:
            self.cols = list(X.columns)
        for col in self.cols:
            X[col] = impute.fit_transform(X[[col]])
        return X

    def fit(self, X):
        return self

# from prediction_model.config import config
# print(config.strategy)
# X = pd.DataFrame({'city':['tokyo', np.nan, 'london', 'seattle', 'sanfrancisco', 'tokyo'],
#           'boolean':['yes', 'no', np.nan, 'no', 'no', 'yes'],
#           'ordinal_column':['somewhat like', 'like', 'somewhat like', 'like',
#                             'somewhat like', 'dislike'],
#           'quantitative_column':[1, 11, -.5, 10, np.nan, 20]})
# cci = CustomImputer(cols=['quantitative_column'],strategy=config.strategy) # here default strategy = mean
# print(cci.fit_transform(X))
