from dataclasses import dataclass
import sys
import numpy as np
import pandas as pd
from typing import Optional
from pandas import DataFrame

from us_visa_prediction.entity.config_entity import USvisaPredictorConfig
from us_visa_prediction.entity.estimator import TargetValueMapping
from us_visa_prediction.cloud_storage.azure_storage import AzureStorageService
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.constants import CURRENT_YEAR
from us_visa_prediction.entity.cloud_estimator import USvisaEstimator

@dataclass
class USvisaData:
    """
    Data class to hold input features for US visa prediction
    """
    continent: str
    education_of_employee: str
    has_job_experience: bool
    requires_job_training: bool
    no_of_employees: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: bool
    yr_of_estab: int

    def get_usvisa_input_data_frame(self) -> DataFrame:
        """
        Convert the input data into a pandas DataFrame
        """
        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "yr_of_estab": [self.yr_of_estab]
            }
            return pd.DataFrame(input_data)
        except Exception as e:
            raise USvisaException(e, sys) from e

class USvisaPredictor:
    def __init__(self, config: USvisaPredictorConfig):
        """
        Initialize the US visa predictor with configuration
        
        Args:
            config: Configuration object containing Azure storage details
        """
        try:
            self.config = config
            self.azure_storage = AzureStorageService(
                container_name=config.container_name
            )
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Create a copy to avoid modifying input
            df = df.copy()
            
            # Add company age
            df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
            
            # Log transformations
            df['log_company_age'] = np.log1p(df['company_age'])
            df['log_no_of_employees'] = np.log1p(df['no_of_employees'])
            
            # Handle wage percentiles for single predictions
            if len(df) == 1:
                df['wage_percentile'] = 0.5  # Use median percentile for single predictions
            else:
                df['wage_percentile'] = df.groupby('continent')['prevailing_wage'].rank(pct=True)
            
            # Drop unnecessary columns
            df.drop(['yr_of_estab'], axis=1, inplace=True, errors='ignore')
            
            logging.info("Completed feature engineering")
            return df
            
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_model(self) -> USvisaEstimator:
        """
        Load the model from Azure storage
        
        Returns:
            Loaded model estimator
        """
        try:
            model_path = self.config.model_path
            
            # Enhanced validation
            if not model_path:
                raise USvisaException("Model path is not configured")
            
            if not self.azure_storage.is_blob_exists(model_path):
                logging.warning(f"Model not found at {model_path}")
                raise USvisaException(f"Model not found at {model_path}")
            
            model = USvisaEstimator(
                container_name=self.config.container_name,
                model_path=model_path
            )
            
            logging.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise USvisaException(e, sys) from e

    def predict(self, input_data: USvisaData) -> dict:
        """
        Make prediction for given input data
        
        Args:
            input_data: USvisaData object containing input features
            
        Returns:
            Dictionary containing prediction and probability
        """
        try:
            # Convert input to DataFrame
            input_df = input_data.get_usvisa_input_data_frame()
            
            # Perform feature engineering
            processed_df = self._perform_feature_engineering(input_df)
            
            # Load model and make prediction
            model = self.get_model()
            prediction = model.predict(processed_df)
            prediction_proba = model.predict_proba(processed_df)
            
            # Convert prediction to human-readable format
            prediction_mapping = {v: k for k, v in TargetValueMapping()._asdict().items()}
            visa_status = prediction_mapping.get(prediction[0], "Unknown")
            probability = float(max(prediction_proba[0]))
            
            return {
                "visa_status": visa_status,
                "probability": probability,
                "status": "success",
                "message": "Prediction completed successfully"
            }
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return {
                "status": "error",
                "message": f"Prediction error: {str(e)}",
                "visa_status": None,
                "probability": None
            }