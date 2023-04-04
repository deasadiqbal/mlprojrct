

import os
import pandas as pd
import sys
from src.exception import ExceptionHandler
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainer
from src.components.model_training import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data_fol', 'train.csv')
    test_data_path: str=os.path.join('data_fol', 'test.csv')
    raw_data_path: str=os.path.join('data_fol', 'raw_data.csv')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def initate_data_ingestion(self):
        logging.info('enter data ingestion')
        try:
            df = pd.read_csv("notebook\data\std.csv")
            logging.info('data ingestion done')
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            train_set, test_set =train_test_split(df, test_size=0.2, random_state=42)
            logging.info('data split done')

            train_set.to_csv(self.config.train_data_path, index=False, header=True)

            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            
            logging.info('data saved successfully') 

            return(
                self.config.train_data_path,
                self.config.test_data_path,
            )
        except Exception as e:
            logging.error('Error while preparing data')
            ExceptionHandler(e) 

if __name__ == '__main__':
    config = DataIngestionConfig()
    obj = DataIngestion(config)
    train_data, test_data = obj.initate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))



'''
    def read_data(self):
        try:
            logging.info('Reading data')
            data = pd.read_csv(self.config.train_data_path)
            logging.info('Data read successfully')
            return data
        except Exception as e:
            logging.error('Error while reading data')
            ExceptionHandler(e)

    def split_data(self, data):
        try:
            logging.info('Splitting data')
            X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=0)
            logging.info('Data split successfully')
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error('Error while splitting data')
            ExceptionHandler(e)

    def save_data(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Saving data')
            X_train.to_csv(self.config.train_data_path, index=False)
            X_test.to_csv(self.config.test_data_path, index=False)
            y_train.to_csv(self.config.train_data_path, index=False)
            y_test.to_csv(self.config.test_data_path, index=False)
            logging.info('Data saved successfully')
        except Exception as e:
            logging.error('Error while saving data')
            ExceptionHandler(e)

    def data_preparation(self):
        try:
            logging.info('Preparing data')
            data = self.read_data()
            X_train, X_test, y_train, y_test = self.split_data(data)
            self.save_data(X_train, X_test, y_train, y_test)
            logging.info('Data prepared successfully')
        except Exception as e:
            logging.error('Error while preparing data')
            ExceptionHandler(e) '''