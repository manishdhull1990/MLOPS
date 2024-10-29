import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline,load_dataset,separate_data
from prediction_model.util.logger_util import logging

classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions():
    logging.info("Triggering from prediction")
    test_data = load_dataset(config.TEST_FILE)
    logging.info("Test data loaded")
    X,y = separate_data(test_data)
    logging.info("Test data separated")
    pred = classification_pipeline.predict(X)
    output = np.where(pred==1,'Diabetic','Not Diabetic')
    logging.info(f"The result is {output}")
    #print(output)
    return output

if __name__=='__main__':
    generate_predictions()