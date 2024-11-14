from pathlib import Path
import sys
import os
from sklearn.model_selection import RandomizedSearchCV
from prediction_model.util.logger_util import logging
import mlflow
import mlflow.sklearn
from sklearn import metrics
from urllib.parse import urlparse
from matplotlib import pyplot as plt

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

class modelFinder:
    def __init__(self,models,param_distributions,experiment_name):
        self.models = models
        self.param_distributions = param_distributions
        self.experiment_name = experiment_name

    def eval_metrics(self,actual,predicted,name):
        accuracy = metrics.accuracy_score(actual,predicted)
        precision = metrics.precision_score(actual,predicted, average='weighted')
        recall = metrics.recall_score(actual,predicted, average='weighted')
        f1 = metrics.f1_score(actual,predicted, average='weighted')
        auc_roc = metrics.roc_auc_score(actual,predicted)
        fpr, tpr, _ = metrics.roc_curve(actual, predicted)
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate', size=14)
        plt.ylabel('True Positive Rate', size=14)
        plt.legend(loc='lower right')
        # Save plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/ROC_curve{name}.png")
        # Close plot
        plt.close()
        return accuracy, precision, recall, f1, auc_roc, auc

    def mlflow_logging(self,model_name,model,param_dist,X_train,X_test,y_train,y_test):
        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()).scheme  # If we dont set a set_tracking_uri, then it returns a file here otherwise http
        print("tracking_url_type_store :", tracking_url_type_store)
        logging.info("Initiate MLFlow for Experiment tracking")
        with mlflow.start_run(run_name=model_name) as run:
            #run_id = run.info.run_id
            #mlflow.set_tag("run_id", run_id)
            random_search = RandomizedSearchCV(
                model,param_dist,n_iter=5,cv=5,scoring='accuracy',random_state=42
            )
            random_search.fit(X_train, y_train)
            best_estimator = random_search.best_estimator_
            best_params = random_search.best_params_
            pred = random_search.predict(X_test)
            #metrics
            accuracy, precision, recall, f1, auc_roc, auc = self.eval_metrics(y_test,pred,model_name)
            # Logging best parameters from RandomSearch
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            # log the metrics
            mlflow.log_metric("Mean CV score", random_search.best_score_)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("f1-score", f1)
            mlflow.log_metric("AUC-ROC score", auc_roc)
            mlflow.log_metric("AUC", auc)

            # Logging artifacts and model
            mlflow.log_artifact(f"plots/ROC_curve{model_name}.png")
            mlflow.sklearn.log_model(model, model_name)
            mlflow.end_run()
            return accuracy, precision, recall, f1, auc_roc, auc, best_estimator

    def bestmodelfinder(self,X_train,X_test,y_train,y_test):
        #print('Finding the best model')
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment(self.experiment_name)
        logging.info('Finding the best model')
        model_report={}
        try:
            for model_name, (model, param_dist) in self.models.items():
                accuracy, precision, recall, f1, auc_roc, auc, best_estimator = self.mlflow_logging(model_name,model,param_dist,X_train,X_test,y_train,y_test)
                model_report[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc_roc': auc_roc,
                        'best_estimator': best_estimator
                    }
                logging.info(f"Best parameters for {model_name} are : {model_report[model_name]}")
            best_model_name = max(model_report, key=lambda k: model_report[k]['auc_roc'])
            best_model = model_report[best_model_name]['best_estimator']
            print(best_model)
            return best_model
        except Exception as e:
            logging.error(f"Error during best model finding: {e}")