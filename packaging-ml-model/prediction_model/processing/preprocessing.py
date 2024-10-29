from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from prediction_model.util.logger_util import logging

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

class CustomNumericalImputer(BaseEstimator,TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
        self.cols = cols
        self.strategy = strategy

    def fit(self, X,y=None):
        return self

    def transform(self, X):
        X=X.copy()
        impute = SimpleImputer(strategy=self.strategy)
        if self.cols == None:
            self.cols = list(X.columns)
        for col in self.cols:
            X[col] = impute.fit_transform(X[[col]])
        #print('Impute',X)
        logging.info("Data imputation is done")
        return X

class CustomScaler(BaseEstimator,TransformerMixin):
    def __init__(self, cols=None,scaling=StandardScaler()):
        self.cols = cols
        self.scaling = scaling

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        scaler = self.scaling
        for col in self.cols:
            X[col] = scaler.fit_transform(X[[col]])
        #print('Scaler',X)
        logging.info("Data scaling is done")
        return X

# from prediction_model.config import config
# print(config.strategy)
# X = pd.DataFrame({'city':['tokyo', np.nan, 'london', 'seattle', 'sanfrancisco', 'tokyo'],
#           'boolean':['yes', 'no', np.nan, 'no', 'no', 'yes'],
#           'ordinal_column':['somewhat like', 'like', 'somewhat like', 'like',
#                             'somewhat like', 'dislike'],
#           'quantitative_column':[1, 11, -.5, 10, np.nan, 20],
#           'population':[100000,200000,300000,250000,500000,100000]        })
# cci = CustomNumericalImputer(['quantitative_column'],config.strategy) # here default strategy = mean
# #print(cci.fit_transform(X))
# df=cci.fit_transform(X)
# scaled_values=CustomScaler(['quantitative_column','population'],config.scaling)
# scaled_values.fit_transform(df)