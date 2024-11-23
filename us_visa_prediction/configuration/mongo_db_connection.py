import os
import sys
from typing import Optional
import certifi
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure

from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.constants import DATABASE_NAME, MONGODB_URL_KEY

class MongoDBClient:
    """
    Handles the connection to the MongoDB database.
    
    This class provides the following functionality:
    - Establishes a connection to the MongoDB database
    - Provides access to the database instance
    - Manages the connection lifecycle
    
    Attributes:
        client: MongoDB client instance
        database: MongoDB database instance
        database_name: Name of the connected MongoDB database
    """

    def __init__(self, database_name: Optional[str] = DATABASE_NAME):
        """
        Initialize the MongoDBClient with the database connection.
        
        Args:
            database_name: Name of the MongoDB database to connect to
            
        Raises:
            USvisaException: If connection initialization fails
        """
        try:
            mongo_db_url = os.getenv(MONGODB_URL_KEY)
            if not mongo_db_url:
                raise ValueError(f"Environment variable '{MONGODB_URL_KEY}' is not set.")

            self.client = MongoClient(mongo_db_url, tlsCAFile=certifi.where())
            self.database: Database = self.client[database_name]
            self.database_name = database_name

            self._validate_connection()
            logging.info(f"Connected to MongoDB database: {self.database_name}")

        except ConnectionFailure as e:
            raise USvisaException(f"Failed to connect to MongoDB: {e}", sys) from e
        except Exception as e:
            raise USvisaException(f"Unexpected error during MongoDB connection: {e}", sys) from e

    def _validate_connection(self) -> None:
        """
        Validate the MongoDB connection is active and working.
        
        Raises:
            USvisaException: If connection validation fails
        """
        try:
            # Ping the database to verify connection
            self.database.command('ping')
        except ConnectionFailure as e:
            raise USvisaException(f"MongoDB connection validation failed: {e}", sys) from e
        except Exception as e:
            raise USvisaException(f"Unexpected error during connection validation: {e}", sys) from e

    def get_database(self) -> Database:
        """
        Get the MongoDB database instance.
        
        Returns:
            MongoDB database instance
        """
        return self.database