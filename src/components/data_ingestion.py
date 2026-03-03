# This is the data ingestion component which is responsible for reading the dataset and splitting it into train and test set. It also saves the train and test set in the artifacts folder.
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# sklearn library is used for train test split
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# The @dataclass decorator is used to create a class that is used to store the configuration for the data ingestion. It has three attributes: train_data_path, test_data_path and raw_data_path. The default value for these attributes is the path where the train, test and raw data will be saved in the artifacts folder.
@dataclass
# The DataIngestionConfig class is used to store the configuration for the data ingestion. It has three attributes: train_data_path, test_data_path and raw_data_path. The default value for these attributes is the path where the train, test and raw data will be saved in the artifacts folder.
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')
    
# The DataIngestion class is responsible for reading the dataset and splitting it into train and test set. It also saves the train and test set in the artifacts folder. The initiate_data_ingestion method is used to read the dataset, split it into train and test set and save it in the artifacts folder. It also returns the path of the train and test set.
class DataIngestion:

    # The __init__ method is used to initialize the DataIngestionConfig class and store it in the ingestion_config attribute. This is done so that we can use the configuration for the data ingestion in the initiate_data_ingestion method.
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # The initiate_data_ingestion method is used to read the dataset, split it into train and test set and save it in the artifacts folder. It also returns the path of the train and test set.  
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset as dataframe
            df = pd.read_csv(os.path.join('notebook/data/stud.csv'))
            logging.info("Read the dataset as dataframe")

            # Create the directory to save the train and test set if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data in the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the dataset into train and test set
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test set in the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                # Return the path of the train and test set
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        # If there is any exception, it will be raised as a CustomException and the error message will be logged in the logs file. The sys module is used to get the information about the exception and the traceback.
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))