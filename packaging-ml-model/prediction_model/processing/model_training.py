from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from sklearn.model_selection import RandomizedSearchCV

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

def bestmodelfinder(X_train,X_test,y_train,y_test):
    print('Finding the best model')
    model_report={}
    for model_name, model in config.models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        model_report[model_name] = score
    print("Model Report:", model_report)
    best_model_name = max(model_report, key=model_report.get)
    best_model =config.models[best_model_name]
    print("Best model is:",best_model)
    if best_model_name in config.param_distributions:
        random_search = RandomizedSearchCV(best_model, config.param_distributions[best_model_name],
                                           n_iter=10, cv=5,
                                           scoring='accuracy',
                                           random_state=42)
        random_search.fit(X_train, y_train)
    print(f"Best parameters for {best_model_name} are : {random_search.best_params_}")
    print(f"Best score for {best_model_name} is: {random_search.best_score_}")
    print(f"Best score for {best_model_name} is: {random_search.best_estimator_}")
    return random_search.best_estimator_
