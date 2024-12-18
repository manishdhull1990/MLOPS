import pathlib
import os
import prediction_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")
LOGPATH = os.path.join(PACKAGE_ROOT,"logs")
LOGFILE = 'log.txt'

FILE_NAME = 'raw.csv'
TRAIN_FILE = 'train.csv'
TEST_FILE = "test.csv"

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'Outcome'
TEST_SIZE = 0.2
#Final features used in the model
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age","Outcome"]
NUMERICAL_COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
                  "BMI","DiabetesPedigreeFunction","Age"]

#Impute Columns
STRATEGY='median'

#Scaling
SCALING=StandardScaler()

#Models
MODELS = {
    'RandomForest': (RandomForestClassifier(),
                     {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}),
    'Ada_Boost': (AdaBoostClassifier(),
                  {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}),
    'LogisticRegression': (LogisticRegression(),
                           {'C':np.logspace(-3,3,7), 'penalty':["l1","l2"]}),
    'SupportVectorMachine': (SVC(),
                        {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']})
}

#parameters
PARAM_DISTRIBUTIONS= {
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
    'LogisticRegression': {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]},
    'SupportVectorMachine': {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
}

EXPERIMENT_NAME="Experiment#1_"+ str(datetime.now().strftime("%d-%m-%y"))
