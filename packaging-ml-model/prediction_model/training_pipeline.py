import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import joblib

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,save_pipeline,separate_data,split_data
import prediction_model.pipeline as pipe
import prediction_model.processing.model_training as training
import sys

def perform_training():
    dataset = load_dataset(config.FILE_NAME)
    X,y = separate_data(dataset)
    transformed_df=pipe.classification_pipeline.fit_transform(X, y)
    X_train,X_test,y_train,y_test = split_data(transformed_df,y)
    train_data = X_train.copy()
    train_data[config.TARGET] = y_train
    #train_data.to_csv(os.path.join(config.DATAPATH, config.TRAIN_FILE))
    test_data = X_test.copy()
    test_data[config.TARGET] = y_test
    #test_data.to_csv(os.path.join(config.DATAPATH,config.TEST_FILE))
    save_pipeline(training.bestmodelfinder(X_train,X_test,y_train,y_test))

if __name__=='__main__':
    print("\n Triggering from training")
    perform_training()