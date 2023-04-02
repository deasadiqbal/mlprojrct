import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import ExceptionHandler
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('data_fol', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Create a data transformer"""
        try:
            num_cols = ['writing score', 'reading score']
            cat_cols = [
                'gender', 'race/ethnicity',
                'parental level of education',
                'lunch', 'test preparation course'
            ]
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical and numerical column transformer created')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_transformer', num_pipeline, num_cols),
                    ('cat_transformer', cat_pipeline, cat_cols)
                ]
            )
            return preprocessor

        except Exception as e:
            logging.error(f'Error in creating data transformer object: {e}')
            raise ExceptionHandler('Error in creating data transformer object')

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test data loaded')

            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()

            target_var = 'math score'
            num_cols = ['writing score', 'reading score']

            training_input_features = train_df.drop(columns=[target_var], axis=1)
            training_target_features = train_df[target_var]

            test_input_features = test_df.drop(columns=[target_var], axis=1)
            test_target_features = test_df[target_var]
            logging.info('Applying preprocessor on train and test data')

            train_feature_arr = preprocessor_obj.fit_transform(training_input_features)
            test_feature_arr = preprocessor_obj.transform(test_input_features)

            train_arr = np.c_[
                train_feature_arr, np.array(training_target_features)
            ]
            test_arr = np.c_[
                test_feature_arr, np.array(test_target_features)
            ]

            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor_obj)
            logging.info('Saved preprocessing object')

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f'Error in data transformation: {e}')
            raise ExceptionHandler('Error in data transformation')
# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('data_fol', 'preprocessor.pkl')


# class DataTransformation:
#     def __init__(self):
#         self.config = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         """This function is used to create a data transformer"""
#         try:
#             num_col = ['writing score', 'reading score']
#             cat_col = [
#                 'gender', 'race/ethnicity',
#                 'parental level of education',
#                 'lunch', 'test preparation course'
#             ]
#             num_pipeline = Pipeline(
#                 steps=[
#                     ('imputer', SimpleImputer(strategy='median')),
#                     ('scaler', StandardScaler(with_mean=False))
#                 ]
#             )
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ('imputer', SimpleImputer(strategy='most_frequent')),
#                     ('onehot', OneHotEncoder(handle_unknown='ignore')),
#                     ('scaler', StandardScaler(with_mean=False))
#                 ]
#             )
#             logging.info('Categorical and numerical column transformer created')

#             preprocessor = ColumnTransformer(
#                 [('num_transformer', StandardScaler(with_mean=False), num_col),
#                  ('cat_transformer', cat_pipeline, cat_col)
#                  ]
#             )
#             return preprocessor
        
# #till here we just build our transformer which will to data transformation

#         except Exception as e:
#             logging.error(f'Error in creating data transformer object: {e}')
#             raise ExceptionHandler('Error in creating data transformer object')
        
# #from here we will perfrome data transformation tecnique

#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)
#             logging.info('Train and test data loaded')

#             logging.info('obtaining preprocessor object')
#             preprocessor_obj = self.get_data_transformer_object()

#             target_var = 'math score'
#             num_col = ['writing score', 'reading score']

#             training_input_feature = train_df.drop(columns=[target_var], axis=1)
#             training_target_feature = train_df[target_var]

#             test_input_feature = test_df.drop(columns=[target_var], axis=1)
#             test_target_feature = test_df[target_var]
#             logging.info('applying preprosessor on train and test data')

#             train_feature_arr = preprocessor_obj.fit_transform(training_input_feature)
#             test_feature_arr = preprocessor_obj.transform(test_input_feature)

#              => np.c_ function numpy arrays ko concatenate karne ke liye istemal hota hai.
#     => np.array(target_feature) target variable ko numpy array mei convert karta hai.
#     => train_arr mei np.c_ function ki madad se train_feature_arr aur target_feature concatenate kiya jata hai,
#        taaki ek hi numpy array mei dono features shamil hojayein

#             train_arr = np.c_[
#                 train_feature_arr, np.array(target_var)
#             ]
#             test_arr = np.c_[
#                 test_feature_arr, np.array(target_var)
#             ]

#             save_object(file_path= self.config.preprocessor_file_path,
#                         obj = preprocessor_obj)              
                
            

#             logging.info('saved preprocessing object')
#             save_object(self.config.preprocessor_obj_file_path, preprocessor_obj)

#             return (train_arr, test_arr, self.config.preprocessor_file_path,)

#         except:
#             logging.error('Error in data transformation')
#             raise ExceptionHandler('Error in data transformation')