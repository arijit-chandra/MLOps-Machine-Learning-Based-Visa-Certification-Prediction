from typing import Optional, List, Dict, Any
import sys
from contextlib import contextmanager

import pandas as pd
import numpy as np
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure

from us_visa_prediction.configuration.mongo_db_connection import MongoDBClient
from us_visa_prediction.constants import DATABASE_NAME
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging


class USvisaData:
    """
    Handles data export operations from MongoDB to pandas DataFrame.
    
    This class provides functionality to:
    - Connect to MongoDB
    - Export collections as DataFrames
    - Handle data cleaning and preprocessing
    - Manage connection lifecycle
    
    Attributes:
        mongo_client: MongoDB client instance
        database: MongoDB database instance
    """

    def __init__(self, database_name: str = DATABASE_NAME):
        """
        Initialize EasyVisaData with MongoDB connection.
        
        Args:
            database_name: Name of the MongoDB database to connect to
            
        Raises:
            USvisaException: If connection initialization fails
        """
        try:
            self.mongo_client = MongoDBClient(database_name=database_name)
            self.database: Database = self.mongo_client.database
            logging.info(f"Connected to MongoDB database: {database_name}")
            
            # Validate database connection
            self._validate_connection()
            
        except Exception as e:
            raise USvisaException(f"Failed to initialize MongoDB connection: {e}", sys) from e

    def _validate_connection(self) -> None:
        """
        Validate MongoDB connection is active and working.
        
        Raises:
            USvisaException: If connection validation fails
        """
        try:
            # Ping database to verify connection
            self.database.command('ping')
        except ConnectionError as e:
            raise USvisaException(f"MongoDB connection validation failed: {e}", sys) from e
        except Exception as e:
            raise USvisaException(f"Unexpected error during connection validation: {e}", sys) from e

    def _get_collection(self, collection_name: str, database_name: Optional[str] = None) -> Collection:
        """
        Get MongoDB collection object with validation.
        
        Args:
            collection_name: Name of the collection
            database_name: Optional alternative database name
            
        Returns:
            MongoDB Collection object
            
        Raises:
            USvisaException: If collection access fails
        """
        try:
            if not collection_name:
                raise ValueError("Collection name cannot be empty")
                
            if database_name:
                collection = self.mongo_client[database_name][collection_name]
            else:
                collection = self.database[collection_name]
                
            # Validate collection exists
            if collection_name not in self.database.list_collection_names():
                raise ValueError(f"Collection '{collection_name}' not found in database")
                
            return collection
            
        except Exception as e:
            raise USvisaException(f"Failed to get collection {collection_name}: {e}", sys) from e

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the DataFrame.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            USvisaException: If cleaning process fails
        """
        try:
            # Remove MongoDB ID column if present
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            
            # Replace "na" strings with np.nan
            df.replace({"na": np.nan}, inplace=True)
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Log cleaning statistics
            logging.info(f"Cleaned DataFrame shape: {df.shape}")
            logging.info(f"Columns with null values: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")
            
            return df
            
        except Exception as e:
            raise USvisaException(f"DataFrame cleaning failed: {e}", sys) from e

    @contextmanager
    def _error_handling_context(self, operation: str):
        """
        Context manager for consistent error handling.
        
        Args:
            operation: Description of the operation being performed
            
        Yields:
            None
            
        Raises:
            USvisaException: If operation fails
        """
        try:
            yield
        except OperationFailure as e:
            raise USvisaException(f"MongoDB operation failed during {operation}: {e}", sys) from e
        except Exception as e:
            raise USvisaException(f"Unexpected error during {operation}: {e}", sys) from e

    def validate_collection_data(self, collection: Collection) -> None:
        """
        Validate collection data meets basic requirements.
        
        Args:
            collection: MongoDB collection to validate
            
        Raises:
            USvisaException: If validation fails
        """
        try:
            # Check if collection is empty
            if collection.count_documents({}) == 0:
                raise ValueError("Collection is empty")
            
            # Add additional validation rules as needed
            # Example: Check for required fields, data types, etc.
            
        except Exception as e:
            raise USvisaException(f"Collection validation failed: {e}", sys) from e

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export entire collection as pandas DataFrame with data cleaning.
        
        Args:
            collection_name: Name of the collection to export
            database_name: Optional alternative database name
            
        Returns:
            pandas DataFrame containing the collection data
            
        Raises:
            USvisaException: If export process fails
        """
        try:
            logging.info(f"Exporting collection: {collection_name}")
            
            with self._error_handling_context("collection export"):
                # Get and validate collection
                collection = self._get_collection(collection_name, database_name)
                self.validate_collection_data(collection)
                
                # Convert to DataFrame
                df = pd.DataFrame(list(collection.find()))
                logging.info(f"Initial DataFrame shape: {df.shape}")
                
                # Clean the DataFrame
                df = self._clean_dataframe(df)
                
                return df
                
        except Exception as e:
            raise USvisaException(f"Failed to export collection as DataFrame: {e}", sys) from e

    def get_collection_names(self, database_name: Optional[str] = None) -> List[str]:
        """
        Get list of all collections in the database.
        
        Args:
            database_name: Optional alternative database name
            
        Returns:
            List of collection names
            
        Raises:
            USvisaException: If operation fails
        """
        try:
            db = self.mongo_client[database_name] if database_name else self.database
            return db.list_collection_names()
        except Exception as e:
            raise USvisaException(f"Failed to get collection names: {e}", sys) from e

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
            
        Raises:
            USvisaException: If operation fails
        """
        try:
            collection = self._get_collection(collection_name)
            return {
                'document_count': collection.count_documents({}),
                'storage_size': collection.stats().get('size', 0),
                'indexes': list(collection.index_information().keys())
            }
        except Exception as e:
            raise USvisaException(f"Failed to get collection stats: {e}", sys) from e