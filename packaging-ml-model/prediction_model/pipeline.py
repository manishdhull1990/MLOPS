from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp
from prediction_model.util.logger_util import logging

class TransformPipeline:
    def __init__(self,numerical_cols,strategy,scaling):
        self.numerical_cols = numerical_cols
        self.strategy = strategy
        self.scaling = scaling

    def transform_classification_pipeline(self):
        try:
            classification_pipeline = Pipeline(steps=
            [
                ('CustomNumericalImputer',pp.CustomNumericalImputer(cols = self.numerical_cols,strategy = self.strategy)),
                ('CustomScaler', pp.CustomScaler(cols = self.numerical_cols,scaling = self.scaling)),
            ]
            )
            return classification_pipeline
        except Exception as e:
            logging.error(f"Error while fitting pipeline: {e}")


# from prediction_model.processing.data_handling import separate_data,load_dataset
#
# dataset = load_dataset(config.FILE_NAME)
# X,y = separate_data(dataset)
# data=classification_pipeline.fit_transform(X,y)
# #print(data['Insulin'].head(),data[['Pregnancies','Glucose','Insulin']].head(),X[['Pregnancies','Glucose','Insulin']].head())
# print(data)