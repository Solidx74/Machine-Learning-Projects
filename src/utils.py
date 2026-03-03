import os
import sys
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.metrics import r2_score

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
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train the model
            y_test_pred = model.predict(X_test)  # Predict on test data
            test_model_score = r2_score(y_test, y_test_pred)  # Calculate r2 score
            report[list(models.keys())[i]] = test_model_score  # Store the score in the report dictionary

        return report

    except Exception as e:
        raise CustomException(e, sys)