from pathlib import Path
import sys
import os
from sklearn.model_selection import RandomizedSearchCV
from prediction_model.util.logger_util import logging

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

class modelFinder:
    def __init__(self,models,param_distributions):
        self.models = models
        self.param_distributions = param_distributions

    def bestmodelfinder(self,X_train,X_test,y_train,y_test):
        #print('Finding the best model')
        logging.info('Finding the best model')
        model_report={}
        try:
            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                model_report[model_name] = score
            #print("Model Report:", model_report)
            logging.info(f"Model Report:{model_report}")
            best_model_name = max(model_report, key=model_report.get)
            best_model =self.models[best_model_name]
            #print("Best model is:",best_model)
            logging.info(f"Best model is:{best_model}")
            if best_model_name in self.param_distributions:
                random_search = RandomizedSearchCV(best_model, self.param_distributions[best_model_name],
                                                   n_iter=10, cv=5,
                                                   scoring='accuracy',
                                                   random_state=42)
                random_search.fit(X_train, y_train)
            #print(f"Best parameters for {best_model_name} are : {random_search.best_params_}")
            #print(f"Best score for {best_model_name} is: {random_search.best_score_}")
            #print(f"Best score for {best_model_name} is: {random_search.best_estimator_}")
            logging.info(f"Best parameters for {best_model_name} are : {random_search.best_params_}")
            logging.info(f"Best score for {best_model_name} is: {random_search.best_score_}")
            logging.info(f"Best score for {best_model_name} is: {random_search.best_estimator_}")
            return random_search.best_estimator_
        except Exception as e:
            logging.error(f"Error during best model finding: {e}")