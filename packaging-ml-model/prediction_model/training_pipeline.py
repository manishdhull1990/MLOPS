from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
import prediction_model.processing.data_handling as data_manipulation
import prediction_model.pipeline as pipeline
import prediction_model.processing.model_training as training
from prediction_model.util.logger_util import logging

class Training:
    def __init__(self,logpath,log_file_name,test_size,datapath, file_name,target,train_file,
                 test_file,model_path,model_name,features,models,param_distributions,
                 numerical_cols,strategy,scaling):
        self.logfilename = os.path.join(logpath,log_file_name)
        self.pipe=pipeline.TransformPipeline(numerical_cols,strategy,scaling).transform_classification_pipeline()
        self.data_manipulation = data_manipulation.data_handling(datapath,file_name,target,
                                                        model_path,model_name,features)
        self.best_model_finder = training.modelFinder(models,param_distributions)
        self.test_size = test_size
        self.datapath = datapath
        self.file_name = file_name
        self.target = target
        self.train_file = train_file
        self.test_file = test_file
        self.features = features

    def perform_training(self):
        try:
            with open(self.logfilename,'w'):pass
            logging.info("Triggering from training")
            dataset = self.data_manipulation.load_dataset()
            X,y = self.data_manipulation.separate_data(dataset)
            logging.info("Data Transformation pipeline has started")
            transformed_df=self.pipe.fit_transform(X, y)
            logging.info("Data Transformation pipeline has finished")
            X_train,X_test,y_train,y_test = self.data_manipulation.split_data(transformed_df,y,self.test_size)
            train_data = X_train.copy()
            train_data[self.target] = y_train
            train_data.to_csv(os.path.join(self.datapath, self.train_file))
            test_data = X_test.copy()
            test_data[self.target] = y_test
            test_data.to_csv(os.path.join(self.datapath,self.test_file))
            self.data_manipulation.save_pipeline(self.best_model_finder.bestmodelfinder(X_train,X_test,y_train,y_test))
        except Exception as e:
            logging.error(f"Error during training: {e}")

# if __name__=='__main__':
#     #print("\n Triggering from training")
#     train=Training(config.LOGPATH,config.LOGFILE,config.TEST_SIZE,config.DATAPATH,
#                    config.FILE_NAME,config.TARGET,config.TRAIN_FILE,config.TEST_FILE,
#                    config.SAVE_MODEL_PATH,config.MODEL_NAME,config.FEATURES,
#                    config.MODELS,config.PARAM_DISTRIBUTIONS,
#                    config.NUMERICAL_COLS,config.STRATEGY,config.SCALING)
#     train.perform_training()