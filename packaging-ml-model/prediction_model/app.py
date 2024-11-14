from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
from prediction_model.config import config
import prediction_model.training_pipeline as training_pipeline
import prediction_model.predict as prediction_pipeline

class classification:
    def __init__(self):
        pass

    def training(self,logpath,log_file_name,test_size,datapath, file_name,target,train_file,
                 test_file,model_path,model_name,features,models,param_distributions,
                 numerical_cols,strategy,scaling,experiment_name):
        train = training_pipeline.Training(logpath,log_file_name,test_size,datapath, file_name,target,train_file,
                 test_file,model_path,model_name,features,models,param_distributions,
                 numerical_cols,strategy,scaling,experiment_name)
        train.perform_training()

    def prediction(self,datapath,logpath, log_file_name,test_file_name,target,model_path,model_name,features):
        predict = prediction_pipeline.Prediction(datapath,logpath, log_file_name,test_file_name,target,model_path,
                                       model_name,features)
        predict.generate_predictions()

if __name__ == '__main__':
    pipeline_execution = input("Enter the training or prediction keyword:")

    cls = classification()
    if pipeline_execution == 'training':
        cls.training(config.LOGPATH,config.LOGFILE,config.TEST_SIZE,config.DATAPATH,
            config.FILE_NAME,config.TARGET,config.TRAIN_FILE,config.TEST_FILE,
            config.SAVE_MODEL_PATH,config.MODEL_NAME,config.FEATURES,config.MODELS,
            config.PARAM_DISTRIBUTIONS,config.NUMERICAL_COLS,config.STRATEGY,
            config.SCALING,config.EXPERIMENT_NAME)

    elif pipeline_execution == 'prediction':
        cls.prediction(config.DATAPATH,config.LOGPATH,config.LOGFILE,config.TEST_FILE,config.TARGET,
                       config.SAVE_MODEL_PATH, config.MODEL_NAME, config.FEATURES)

    else:
        print("Wrong keyword given")