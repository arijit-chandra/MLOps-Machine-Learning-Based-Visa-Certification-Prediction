# us_visa_data.py
import sys
from typing import Dict, Union
import pandas as pd
from pandas import DataFrame
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
# from us_visa_prediction.utils.main_utils import read_yaml_file
from us_visa_prediction.entity.cloud_estimator import USvisaEstimator
from dataclasses import dataclass
from typing import Optional

@dataclass
class USvisaData:
    """
    Data class for US visa prediction input features
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
    company_age: int

    def get_usvisa_input_data_frame(self) -> DataFrame:
        """
        Convert input data to DataFrame
        
        Returns:
            DataFrame: Input data as pandas DataFrame
        
        Raises:
            USvisaException: If conversion fails
        """
        try:
            input_dict = self._get_usvisa_data_as_dict()
            return pd.DataFrame(input_dict)
        except Exception as e:
            logging.error(f"Failed to create DataFrame from input data: {str(e)}")
            raise USvisaException(e, sys) from e

    def _get_usvisa_data_as_dict(self) -> Dict[str, list]:
        """
        Convert input data to dictionary format
        
        Returns:
            Dict[str, list]: Input data as dictionary
        
        Raises:
            USvisaException: If conversion fails
        """
        try:
            return {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }
        except Exception as e:
            logging.error(f"Failed to convert input data to dictionary: {str(e)}")
            raise USvisaException(e, sys) from e

class USvisaClassifier:
    """
    Classifier for US visa prediction using cloud-stored model
    """
    def __init__(self, prediction_pipeline_config: Optional[Dict] = None) -> None:
        """
        Initialize classifier with configuration
        
        Args:
            prediction_pipeline_config: Configuration for prediction pipeline
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config or {}
            self.model_container = self.prediction_pipeline_config.get('model_container_name', 'models')
            self.model_path = self.prediction_pipeline_config.get('model_file_path', 'visa_model.pkl')
            
            logging.info(f"Initialized USvisaClassifier with container: {self.model_container}")
            
            self.estimator = USvisaEstimator(
                container_name=self.model_container,
                model_path=self.model_path
            )
        except Exception as e:
            logging.error(f"Failed to initialize USvisaClassifier: {str(e)}")
            raise USvisaException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> Union[str, list]:
        """
        Make prediction using the model
        
        Args:
            dataframe: Input data for prediction
            
        Returns:
            Union[str, list]: Prediction result
            
        Raises:
            USvisaException: If prediction fails
        """
        try:
            logging.info("Starting prediction process")
            
            if not isinstance(dataframe, DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if dataframe.empty:
                raise ValueError("Input DataFrame is empty")
                
            result = self.estimator.predict(dataframe)
            logging.info("Prediction completed successfully")
            
            return result
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise USvisaException(e, sys) from e

    def validate_inputs(self, dataframe: DataFrame) -> None:
        """
        Validate input data before prediction
        
        Args:
            dataframe: Input data to validate
            
        Raises:
            USvisaException: If validation fails
        """
        required_columns = {
            "continent", "education_of_employee", "has_job_experience",
            "requires_job_training", "no_of_employees", "region_of_employment",
            "prevailing_wage", "unit_of_wage", "full_time_position", "company_age"
        }
        
        try:
            # Check for missing required columns
            missing_columns = required_columns - set(dataframe.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate data types
            if not isinstance(dataframe['prevailing_wage'].iloc[0], (int, float)):
                raise ValueError("prevailing_wage must be numeric")
                
            if not isinstance(dataframe['no_of_employees'].iloc[0], (int)):
                raise ValueError("no_of_employees must be integer")
                
            if not isinstance(dataframe['company_age'].iloc[0], (int)):
                raise ValueError("company_age must be integer")
                
            logging.info("Input validation completed successfully")
            
        except Exception as e:
            logging.error(f"Input validation failed: {str(e)}")
            raise USvisaException(e, sys) from e