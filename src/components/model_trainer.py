import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains each model and evaluates r2_score on test data.
    Returns a dictionary with model names and scores.
    """
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


# Used to store the configuration for the model trainer.
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )



        # The models to be trained are defined in the dictionary.
        # The key of the dictionary is the name of the model and the value is the instance of the model. 
        # The models to be trained are Random Forest, Decision Tree, Gradient Boosting, Linear Regression, K-Neighbors Regressor, XGBRegressor and CatBoosting Regressor.
        # The best model is selected based on the r2_score and the best model is saved in the artifacts folder. The path of the saved model is returned.

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
        # The model_report dictionary is used to store the r2_score of each model. 
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                # Predicting the test set results
                y_test_pred = model.predict(X_test)

                # Evaluating the model
                model_report[list(models.keys())[i]] = r2_score(y_test, y_test_pred)

        # Selecting the best model based on the r2_score.
            best_model_score = max(sorted(model_report.values()))
        # Best model name is selected based on the best model score.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with r2_score greater than 0.6")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
