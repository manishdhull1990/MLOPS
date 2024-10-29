import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.util.logger_util import logging

class data_handling:
    def __init__(self,datapath, raw_file_name,target,model_path,model_name,features):
        self.datapath = datapath
        self.raw_file_name = raw_file_name
        self.filepath = os.path.join(datapath, raw_file_name)
        self.target = target
        self.model_path = model_path
        self.model_name = model_name
        self.features = features

    #Load the dataset
    def load_dataset(self):
        try:
            _data = pd.read_csv(self.filepath)
            logging.info(f"File {self.raw_file_name} loaded in the dataframe from path {self.datapath}")
            return _data[self.features]
        except Exception as e:
            logging.error(f"Error during loading of dataset: {e}")

    # Separate X and y
    def separate_data(self,data):
        try:
            X = data.drop(self.target, axis=1)
            y= data[self.target]
            logging.info(f"Data got separated in target column {self.target} and independent columns {X.columns}")
            return X,y
        except Exception as e:
            logging.error(f"Error during separation of dataset: {e}")

    # Split into training and testing sets
    def split_data(self,X, y, test_size=0.2, random_state=42):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            logging.info("Data is splitted into train and test")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error during splitting of dataset: {e}")

    #Serialization
    def save_pipeline(self,pipeline_to_save):
        try:
            save_path = os.path.join(self.model_path,self.model_name)
            #print(save_path)
            joblib.dump(pipeline_to_save, save_path)
            logging.info(f"Model has been saved under the name {self.model_name}")
            #print(f"Model has been saved under the name {config.MODEL_NAME}")
        except Exception as e:
            logging.error(f"Error during saving model: {e}")

    #Deserialization
    def load_pipeline(self):
        try:
            save_path = os.path.join(self.model_path,self.model_name)
            model_loaded = joblib.load(save_path)
            #print(f"Model has been loaded")
            logging.info(f"Model has been loaded")
            return model_loaded
        except Exception as e:
            logging.error(f"Error during loading model: {e}")