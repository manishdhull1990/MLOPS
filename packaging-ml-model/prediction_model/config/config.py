import pathlib
import os
import prediction_model
PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

FILE_NAME = 'train.csv'
TEST_FILE = "test.csv"

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'Outcome'

#Final features used in the model
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age","Outcome"]
NUMERICAL_COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
                  "BMI","DiabetesPedigreeFunction","Age"]

#Impute Columns
strategy='median'