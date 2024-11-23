import os
import sys
from datetime import date
from typing import Final

from us_visa_prediction.logger import logging

try:
    # Database related constants
    DATABASE_NAME: Final[str] = "EASY_VISA"
    COLLECTION_NAME: Final[str] = "visa_data"
    MONGODB_URL_KEY: Final[str] = "MONGODB_URL"

    # Pipeline basic configuration
    PIPELINE_NAME: Final[str] = "usvisa"
    ARTIFACT_DIR: Final[str] = "artifact"
    CURRENT_YEAR: Final[int] = date.today().year

    # File names and paths
    TRAIN_FILE_NAME: Final[str] = "train.csv"
    TEST_FILE_NAME: Final[str] = "test.csv"
    FILE_NAME: Final[str] = "usvisa.csv"
    MODEL_FILE_NAME: Final[str] = "model.pkl"
    PREPROCSSING_OBJECT_FILE_NAME: Final[str] = "preprocessing.pkl"
    TARGET_COLUMN: Final[str] = "case_status"
    SCHEMA_FILE_PATH: Final[str] = os.path.join("config", "schema.yaml")

    # Data Ingestion Configuration
    DATA_INGESTION_COLLECTION_NAME: Final[str] = "visa_data"
    DATA_INGESTION_DIR_NAME: Final[str] = "data_ingestion"
    DATA_INGESTION_FEATURE_STORE_DIR: Final[str] = "feature_store"
    DATA_INGESTION_INGESTED_DIR: Final[str] = "ingested"
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: Final[float] = 0.2

    # Data Validation Configuration
    DATA_VALIDATION_DIR_NAME: Final[str] = "data_validation"
    DATA_VALIDATION_DRIFT_REPORT_DIR: Final[str] = "drift_report"
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: Final[str] = "report.yaml"

    # Data Transformation Configuration
    DATA_TRANSFORMATION_DIR_NAME: Final[str] = "data_transformation"
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: Final[str] = "transformed"
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: Final[str] = "transformed_object"

    # Model Trainer Configuration
    MODEL_TRAINER_DIR_NAME: Final[str] = "model_trainer"
    MODEL_TRAINER_TRAINED_MODEL_DIR: Final[str] = "trained_model"
    MODEL_TRAINER_TRAINED_MODEL_NAME: Final[str] = "model.pkl"
    MODEL_TRAINER_EXPECTED_SCORE: Final[float] = 0.6
    MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: Final[str] = os.path.join("config", "model.yaml")

    # Model Evaluation Configuration
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: Final[float] = 0.07

    # Azure Blob Storage Configuration
    MODEL_CONTAINER_NAME: Final[str] = "us-visa-model-container"
    AZURE_STORAGE_CONNECTION_STRING: Final[str] = "AZURE_STORAGE_CONNECTION_STRING"

    # Application Configuration
    APP_HOST: Final[str] = "0.0.0.0"
    APP_PORT: Final[int] = 8080

except Exception as e:
    raise Exception(e) from e

def validate_configurations() -> None:
    """
    Validate all configuration settings.
    
    Raises:
        Exception: If any configuration validation fails
    """
    try:
        logging.info("Validating configurations...")
        
        # Validate file paths
        if not os.path.exists(SCHEMA_FILE_PATH):
            raise Exception(f"Schema file not found at {SCHEMA_FILE_PATH}")
            
        if not os.path.exists(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH):
            raise Exception(f"Model config file not found at {MODEL_TRAINER_MODEL_CONFIG_FILE_PATH}")
        
        # Validate numeric values
        if not 0 < DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO < 1:
            raise Exception(f"Invalid train-test split ratio: {DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO}")
            
        if not 0 <= MODEL_TRAINER_EXPECTED_SCORE <= 1:
            raise Exception(f"Invalid expected score: {MODEL_TRAINER_EXPECTED_SCORE}")
            
        if not 0 <= MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE <= 1:
            raise Exception(f"Invalid threshold score: {MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE}")
            
        if not 0 <= APP_PORT <= 65535:
            raise Exception(f"Invalid port number: {APP_PORT}")
        
        logging.info("All configurations validated successfully")
    except Exception as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        raise Exception(e) from e

def create_directories() -> None:
    """
    Create necessary directories for the pipeline.
    
    Raises:
        Exception: If directory creation fails
    """
    try:
        directories = [
            os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME),
            os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DIR_NAME),
            os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME),
            os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME),
            os.path.join(ARTIFACT_DIR, DATA_INGESTION_FEATURE_STORE_DIR),
            os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DRIFT_REPORT_DIR),
            os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR),
            os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR),
            os.path.join(ARTIFACT_DIR, MODEL_TRAINER_TRAINED_MODEL_DIR)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")
            
    except Exception as e:
        logging.error(f"Failed to create directories: {str(e)}")
        raise Exception(e) from e

# Initialize configurations on module import
try:
    logging.info("Initializing configurations...")
    validate_configurations()
    create_directories()
    logging.info("Configuration initialization completed successfully")
except Exception as e:
    logging.critical(f"Failed to initialize configurations: {str(e)}")
    raise Exception(e) from e
