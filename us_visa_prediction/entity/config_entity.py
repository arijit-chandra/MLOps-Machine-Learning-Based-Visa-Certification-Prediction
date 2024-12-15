### config_entity.py file
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Final, Optional
from pathlib import Path

from us_visa_prediction.constants import *
from us_visa_prediction.logger import logging
from us_visa_prediction.exception import USvisaException

TIMESTAMP: Final[str] = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass(frozen=True)
class TrainingPipelineConfig:
    """
    Configuration for the main training pipeline.
    Manages the overall pipeline settings and artifact directory.
    """
    pipeline_name: str = field(default=PIPELINE_NAME)
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.path.join(os.getcwd(), ARTIFACT_DIR)) / TIMESTAMP
    )
    timestamp: str = field(default=TIMESTAMP)

    def __post_init__(self):
        """
        Ensure artifact directory exists after initialization.
        Creates the directory with full path support.
        """
        try:
            # Ensure artifact directory is created
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created artifact directory: {self.artifact_dir}")
        except Exception as e:
            logging.error(f"Failed to create artifact directory: {e}")
            raise USvisaException(f"Failed to create artifact directory: {e}") from e

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion phase.
    Manages paths, collection, and train-test split settings.
    """
    training_pipeline_config: TrainingPipelineConfig = field(
        default_factory=TrainingPipelineConfig
    )
    data_ingestion_dir: Path = field(init=False)
    feature_store_file_path: Path = field(init=False)
    training_file_path: Path = field(init=False)
    testing_file_path: Path = field(init=False)
    train_test_split_ratio: float = field(default=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO)
    collection_name: str = field(default=DATA_INGESTION_COLLECTION_NAME)

    def __post_init__(self):
        """
        Initialize dependent paths and create necessary directories.
        Uses object.__setattr__ due to frozen dataclass.
        """
        try:
            # Set paths with proper directory structure
            object.__setattr__(self, 'data_ingestion_dir', 
                             self.training_pipeline_config.artifact_dir / DATA_INGESTION_DIR_NAME)
            
            object.__setattr__(self, 'feature_store_file_path', 
                             self.data_ingestion_dir / DATA_INGESTION_FEATURE_STORE_DIR / FILE_NAME)
            
            object.__setattr__(self, 'training_file_path',
                             self.data_ingestion_dir / DATA_INGESTION_INGESTED_DIR / TRAIN_FILE_NAME)
            
            object.__setattr__(self, 'testing_file_path',
                             self.data_ingestion_dir / DATA_INGESTION_INGESTED_DIR / TEST_FILE_NAME)
            
            # Create necessary directories
            directories = [
                self.data_ingestion_dir, 
                self.feature_store_file_path.parent, 
                self.training_file_path.parent
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Data ingestion directories created successfully")
        except Exception as e:
            logging.error(f"Error in DataIngestionConfig initialization: {e}")
            raise USvisaException(f"Error in DataIngestionConfig initialization: {e}") from e

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for data validation phase.
    Manages validation directory and drift report settings.
    """
    training_pipeline_config: TrainingPipelineConfig = field(
        default_factory=TrainingPipelineConfig
    )
    data_validation_dir: Path = field(init=False)
    drift_report_file_path: Path = field(init=False)

    def __post_init__(self):
        """
        Initialize validation paths and create necessary directories.
        """
        try:
            object.__setattr__(self, 'data_validation_dir',
                             self.training_pipeline_config.artifact_dir / DATA_VALIDATION_DIR_NAME)
            
            object.__setattr__(self, 'drift_report_file_path',
                             self.data_validation_dir / DATA_VALIDATION_DRIFT_REPORT_DIR / 
                             DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
            
            # Create directories
            directories = [
                self.data_validation_dir, 
                self.drift_report_file_path.parent
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Data validation directories created successfully")
        except Exception as e:
            logging.error(f"Error in DataValidationConfig initialization: {e}")
            raise USvisaException(f"Error in DataValidationConfig initialization: {e}") from e

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for data transformation phase.
    Manages paths for transformed data and preprocessing objects.
    """
    training_pipeline_config: TrainingPipelineConfig = field(
        default_factory=TrainingPipelineConfig
    )
    data_transformation_dir: Path = field(init=False)
    transformed_train_file_path: Path = field(init=False)
    transformed_test_file_path: Path = field(init=False)
    transformed_object_file_path: Path = field(init=False)

    def __post_init__(self):
        """
        Initialize transformation paths and create necessary directories.
        """
        try:
            object.__setattr__(self, 'data_transformation_dir',
                             self.training_pipeline_config.artifact_dir / DATA_TRANSFORMATION_DIR_NAME)
            
            object.__setattr__(self, 'transformed_train_file_path',
                             self.data_transformation_dir / DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR /
                             TRAIN_FILE_NAME.replace("csv", "npy"))
            
            object.__setattr__(self, 'transformed_test_file_path',
                             self.data_transformation_dir / DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR /
                             TEST_FILE_NAME.replace("csv", "npy"))
            
            object.__setattr__(self, 'transformed_object_file_path',
                             self.data_transformation_dir / DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR /
                             PREPROCSSING_OBJECT_FILE_NAME)
            
            # Create directories
            directories = [
                self.data_transformation_dir, 
                self.transformed_train_file_path.parent, 
                self.transformed_object_file_path.parent
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Data transformation directories created successfully")
        except Exception as e:
            logging.error(f"Error in DataTransformationConfig initialization: {e}")
            raise USvisaException(f"Error in DataTransformationConfig initialization: {e}") from e

@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for model training phase.
    Manages model training directory, expected accuracy, and model configuration.
    """
    training_pipeline_config: TrainingPipelineConfig = field(
        default_factory=TrainingPipelineConfig
    )
    model_trainer_dir: Path = field(init=False)
    trained_model_file_path: Path = field(init=False)
    expected_accuracy: float = field(default=MODEL_TRAINER_EXPECTED_SCORE)
    model_config_file_path: Path = field(default=Path(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH))

    def __post_init__(self):
        """
        Initialize model training paths and create necessary directories.
        """
        try:
            object.__setattr__(self, 'model_trainer_dir',
                             self.training_pipeline_config.artifact_dir / MODEL_TRAINER_DIR_NAME)
            
            object.__setattr__(self, 'trained_model_file_path',
                             self.model_trainer_dir / MODEL_TRAINER_TRAINED_MODEL_DIR / MODEL_FILE_NAME)
            
            # Create directories
            directories = [
                self.model_trainer_dir, 
                self.trained_model_file_path.parent
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Model trainer directories created successfully")
        except Exception as e:
            logging.error(f"Error in ModelTrainerConfig initialization: {e}")
            raise USvisaException(f"Error in ModelTrainerConfig initialization: {e}") from e

@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration for model evaluation phase.
    Manages model evaluation thresholds and paths.
    """
    training_pipeline_config: TrainingPipelineConfig = field(
        default_factory=TrainingPipelineConfig
    )
    changed_threshold_score: float = field(default=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE)
    container_name: str = field(default=MODEL_CONTAINER_NAME)
    blob_model_key_path: str = field(default=MODEL_FILE_NAME)
    model_path: Path = field(init=False)

    def __post_init__(self):
        """
        Initialize model evaluation paths and create necessary directories.
        """
        try:
            object.__setattr__(self, 'model_path',
                             self.training_pipeline_config.artifact_dir / 
                             MODEL_TRAINER_DIR_NAME / MODEL_TRAINER_TRAINED_MODEL_DIR / MODEL_FILE_NAME)
            
            # Ensure model directory exists
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Created model evaluation path: {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to create model evaluation path: {e}")
            raise USvisaException(f"Failed to create model evaluation path: {e}") from e

@dataclass(frozen=True)
class ModelPusherConfig:
    """
    Configuration for model pushing phase.
    Manages Azure Blob Storage container and model path details.
    """
    container_name: str = field(default=MODEL_CONTAINER_NAME)
    blob_model_path: str = field(default=MODEL_FILE_NAME)

@dataclass(frozen=True)
class USvisaPredictorConfig:
    """
    Configuration for US visa prediction service.
    Manages model file and container details for prediction.
    """
    model_file_path: str = field(default=MODEL_FILE_NAME)
    model_container_name: str = field(default=MODEL_CONTAINER_NAME)

def validate_configurations() -> None:
    """
    Validate all configuration settings and parameters.
    
    Raises:
        Exception: If any configuration validation fails
    """
    try:
        logging.info("Validating configurations...")
        
        # Validate file paths
        config_files = [
            SCHEMA_FILE_PATH,
            MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
        ]
        
        for file_path in config_files:
            if not os.path.exists(file_path):
                raise Exception(f"Configuration file not found: {file_path}")
        
        # Validate numeric values
        numeric_configs = [
            (DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO, "Train-test split ratio", 0, 1),
            (MODEL_TRAINER_EXPECTED_SCORE, "Expected model score", 0, 1),
            (MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE, "Model evaluation threshold", 0, 1)
        ]
        
        for value, name, min_val, max_val in numeric_configs:
            if not min_val <= value <= max_val:
                raise Exception(f"Invalid {name}: {value}. Must be between {min_val} and {max_val}")
        
        # Validate port number
        if not 0 <= APP_PORT <= 65535:
            raise Exception(f"Invalid port number: {APP_PORT}")
        
        logging.info("All configurations validated successfully")
    except Exception as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        raise USvisaException(f"Configuration validation failed: {str(e)}") from e

# Validate configurations when the module is imported
try:
    validate_configurations()
    logging.info("Configuration module initialized successfully")
except Exception as e:
    logging.critical(f"Failed to initialize configurations: {str(e)}")
    raise