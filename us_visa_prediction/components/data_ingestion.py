### This is data_ingestion.py file
from pathlib import Path
from typing import Tuple, Optional
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from us_visa_prediction.entity.config_entity import DataIngestionConfig
from us_visa_prediction.entity.artifact_entity import DataIngestionArtifact
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.data_access.easyvisa_data import USvisaData


@dataclass
class DataFrameStats:
    """Statistics about the DataFrame for validation and logging purposes."""
    total_rows: int
    total_columns: int
    missing_values: dict
    duplicate_rows: int


class DataIngestion:
    """Handles the data ingestion process including data extraction and train-test splitting."""
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            data_ingestion_config: Configuration for data ingestion process
            
        Raises:
            USvisaException: If initialization fails
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self._validate_config()
        except Exception as e:
            raise USvisaException(f"Failed to initialize DataIngestion: {e}", sys) from e

    def _validate_config(self) -> None:
        """
        Validate the data ingestion configuration.
        
        Raises:
            USvisaException: If configuration validation fails
        """
        try:
            if not isinstance(self.data_ingestion_config, DataIngestionConfig):
                raise USvisaException("data_ingestion_config must be an instance of DataIngestionConfig")
            
            if not 0 < self.data_ingestion_config.train_test_split_ratio < 1:
                raise USvisaException("train_test_split_ratio must be between 0 and 1")
            
            if not self.data_ingestion_config.collection_name:
                raise USvisaException("collection_name cannot be empty")
                
        except Exception as e:
            raise USvisaException(f"Config validation failed: {e}", sys) from e

    def _analyze_dataframe(self, df: pd.DataFrame) -> DataFrameStats:
        """
        Analyze the DataFrame and return key statistics.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            DataFrameStats object containing DataFrame statistics
        """
        return DataFrameStats(
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=df.isnull().sum().to_dict(),
            duplicate_rows=df.duplicated().sum()
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate the DataFrame for basic quality checks.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            USvisaException: If validation fails
        """
        try:
            if df.empty:
                raise USvisaException("DataFrame is empty")
            
            stats = self._analyze_dataframe(df)
            logging.info(f"DataFrame statistics: {stats}")
            
            # Add specific validation rules based on business requirements
            missing_threshold = 0.2  # 20% missing values threshold
            for column, missing_count in stats.missing_values.items():
                if missing_count / stats.total_rows > missing_threshold:
                    logging.warning(f"Column {column} has {missing_count} missing values")
                    
        except Exception as e:
            raise USvisaException(f"DataFrame validation failed: {e}", sys) from e

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export data from MongoDB to feature store.
        
        Returns:
            DataFrame containing the exported data
            
        Raises:
            USvisaException: If export process fails
        """
        try:
            logging.info("Starting data export from MongoDB")
            usvisa_data = USvisaData()
            
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            
            self._validate_dataframe(dataframe)
            
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            feature_store_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Saving data to feature store: {feature_store_path}")
            dataframe.to_csv(feature_store_path, index=False)
            
            return dataframe

        except Exception as e:
            raise USvisaException(f"Data export failed: {e}", sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into train and test sets.
        
        Args:
            dataframe: Input DataFrame to split
            
        Returns:
            Tuple containing train and test DataFrames
            
        Raises:
            USvisaException: If splitting process fails
        """
        try:
            logging.info("Starting train-test split")
            
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42  # Set for reproducibility
            )
            
            # Save split datasets
            for data, path in [
                (train_set, self.data_ingestion_config.training_file_path),
                (test_set, self.data_ingestion_config.testing_file_path)
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                data.to_csv(path, index=False)
                logging.info(f"Saved split data to: {path}")
            
            return train_set, test_set

        except Exception as e:
            raise USvisaException(f"Train-test split failed: {e}", sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiate the data ingestion process.
        
        Returns:
            DataIngestionArtifact containing paths to train and test sets
            
        Raises:
            USvisaException: If data ingestion process fails
        """
        try:
            logging.info("Initiating data ingestion")
            
            # Export data from MongoDB to feature store
            dataframe = self.export_data_into_feature_store()
            logging.info(f"Exported data shape: {dataframe.shape}")
            
            # Split data into train and test sets
            train_set, test_set = self.split_data_as_train_test(dataframe)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            # Create and validate artifact
            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data ingestion completed successfully: {artifact}")
            return artifact

        except Exception as e:
            raise USvisaException(f"Data ingestion failed: {e}", sys) from e