import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from us_visa_prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa_prediction.entity.config_entity import DataTransformationConfig
from us_visa_prediction.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa_prediction.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            num_features = self._schema_config['num_features']

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def _perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering steps including log transforms and percentiles"""
        try:
            # Add company age
            df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
            
            # Log transformations
            df['log_company_age'] = np.log1p(df['company_age'])
            df['log_no_of_employees'] = np.log1p(df['no_of_employees'])
            
            # Wage percentiles by continent
            df['wage_percentile'] = df.groupby('continent')['prevailing_wage'].rank(pct=True)
            
            logging.info("Added engineered features: log transforms and wage percentiles")
            return df
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Perform feature engineering on both datasets
                train_df = self._perform_feature_engineering(train_df)
                test_df = self._perform_feature_engineering(test_df)

                # Before dropping
                print("Train dataset shape before dropping NaNs:", train_df.shape)
                train_df = train_df.dropna()
                print("Train dataset shape after dropping NaNs:", train_df.shape)

                print("Test dataset shape before dropping NaNs:", test_df.shape)
                test_df = test_df.dropna()
                print("Test dataset shape after dropping NaNs:", test_df.shape)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Got train features and test features")

                drop_cols = self._schema_config['drop_columns']
                logging.info("Dropping specified columns")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
                input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
                
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )
                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )

                logging.info("Applying preprocessing object on training and testing datasets")
                
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")
                
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                logging.info("Created train array and test array")
                
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e, sys) from e