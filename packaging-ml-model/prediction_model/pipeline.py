from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.processing.preprocessing as pp

classification_pipeline = Pipeline(steps=
    [
        ('CustomNumericalImputer',pp.CustomNumericalImputer(cols = config.NUMERICAL_COLS,strategy = config.strategy)),
        ('CustomScaler', pp.CustomScaler(cols = config.NUMERICAL_COLS,scaling = config.scaling))
    ]
)

# from prediction_model.processing.data_handling import separate_data,load_dataset
#
# dataset = load_dataset(config.FILE_NAME)
# X,y = separate_data(dataset)
# data=classification_pipeline.fit_transform(X,y)
# #print(data['Insulin'].head(),data[['Pregnancies','Glucose','Insulin']].head(),X[['Pregnancies','Glucose','Insulin']].head())
# print(data)