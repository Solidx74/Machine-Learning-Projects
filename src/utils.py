
import sys
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
# Used to save the preprocessor object in the artifacts folder.
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for name, model in models.items():
            model_param = param[name]

            gs = GridSearchCV(model, model_param, cv=3, scoring='r2')
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            trained_models[name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with(open(file_path, 'rb')) as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

