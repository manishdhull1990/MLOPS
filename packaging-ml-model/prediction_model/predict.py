import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.data_handling as data_manipulation
from prediction_model.util.logger_util import logging

class Prediction:
    def __init__(self,datapath,logpath,log_file_name,test_file_name,target,model_path,model_name,features):
        self.data_manipulation = data_manipulation.data_handling(datapath, test_file_name, target,
                                                                 model_path, model_name, features)
        self.logfilename = os.path.join(logpath,log_file_name)

    def generate_predictions(self):
        try:
            logging.info("Triggering from prediction")
            self.classification_pipeline = self.data_manipulation.load_pipeline()
            test_data = self.data_manipulation.load_dataset()
            X,y = self.data_manipulation.separate_data(test_data)
            pred = self.classification_pipeline.predict(X)
            output = np.where(pred==1,'Diabetic','Not Diabetic')
            logging.info(f"The result is {output}")
            #print(output)
            return output
        except Exception as e:
            logging.error(f"Error in prediction: {e}")

# if __name__=='__main__':
#     predict=Prediction(config.DATAPATH,config.LOGFILE,config.TEST_FILE,config.TARGET,
#                        config.SAVE_MODEL_PATH, config.MODEL_NAME, config.FEATURES)
#     predict.generate_predictions()