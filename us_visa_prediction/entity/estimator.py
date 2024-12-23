import sys
import numpy as np
import pandas as pd
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
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def _perform_feature_engineering(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        # Convert to numeric types first
        df['company_age'] = pd.to_numeric(df['company_age'])
        df['no_of_employees'] = pd.to_numeric(df['no_of_employees'])
        df['prevailing_wage'] = pd.to_numeric(df['prevailing_wage'])
        
        df['log_company_age'] = np.log1p(df['company_age'])
        df['log_no_of_employees'] = np.log1p(df['no_of_employees'])
        df['wage_percentile'] = df.groupby('continent')['prevailing_wage'].rank(pct=True)
        return df

    def predict(self, dataframe: DataFrame):
        try:
            dataframe = self._perform_feature_engineering(dataframe)
            
            numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = dataframe.select_dtypes(include=['object']).columns
            
            for col in numeric_columns:
                if dataframe[col].isna().any():
                    dataframe[col].fillna(dataframe[col].median(), inplace=True)
            
            for col in categorical_columns:
                if dataframe[col].isna().any():
                    dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)

            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed_feature)
            
        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"