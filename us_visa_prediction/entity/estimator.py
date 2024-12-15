import sys
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 0
        self.Denied: int = 1
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of USvisaModel class")
        try:
            logging.info("Checking and handling NaN values")
            
            # Check for NaN values
            nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
            if nan_columns:
                logging.warning(f"NaN values found in columns: {nan_columns}")
                
                # Create a copy of the dataframe to avoid modifying the original
                dataframe = dataframe.copy()
                
                # Impute numeric columns with median
                numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_columns:
                    if dataframe[col].isna().any():
                        logging.info(f"Imputing NaN values in column {col} with median")
                        dataframe[col].fillna(dataframe[col].median(), inplace=True)
                
                # Impute categorical columns with mode
                categorical_columns = dataframe.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    if dataframe[col].isna().any():
                        logging.info(f"Imputing NaN values in column {col} with mode")
                        dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
            
            logging.info("Using the trained model to get predictions")
            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)
        
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"