import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Final
from pathlib import Path

from us_visa_prediction.constants import *
from us_visa_prediction.logger import logging
from us_visa_prediction.exception import USvisaException

TIMESTAMP: Final[str] = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass(frozen=True)
class TrainingPipelineConfig:
    """Configuration for the main training pipeline."""
    pipeline_name: str = field(default=PIPELINE_NAME)
    artifact_dir: Path = field(default_factory=lambda: Path(ARTIFACT_DIR) / TIMESTAMP)
    timestamp: str = field(default=TIMESTAMP)

    def __post_init__(self):
        """Ensure artifact directory exists after initialization."""
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created artifact directory: {self.artifact_dir}")
        except Exception as e:
            raise USvisaException(f"Failed to create artifact directory: {e}") from e

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion phase."""
    data_ingestion_dir: Path = field(
        default_factory=lambda: training_pipeline_config.artifact_dir / DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: Path = field(init=False)
    training_file_path: Path = field(init=False)
    testing_file_path: Path = field(init=False)
    train_test_split_ratio: float = field(default=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO)
    collection_name: str = field(default=DATA_INGESTION_COLLECTION_NAME)

    def __post_init__(self):
        """Initialize dependent paths and create directories."""
        try:
            # Using object.__setattr__ because the dataclass is frozen
            object.__setattr__(self, 'feature_store_file_path', 
                             self.data_ingestion_dir / DATA_INGESTION_FEATURE_STORE_DIR / FILE_NAME)
            object.__setattr__(self, 'training_file_path',
                             self.data_ingestion_dir / DATA_INGESTION_INGESTED_DIR / TRAIN_FILE_NAME)
            object.__setattr__(self, 'testing_file_path',
                             self.data_ingestion_dir / DATA_INGESTION_INGESTED_DIR / TEST_FILE_NAME)
            
            # Create necessary directories
            self.data_ingestion_dir.mkdir(parents=True, exist_ok=True)
            self.feature_store_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.training_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Data ingestion directories created successfully")
        except Exception as e:
            raise USvisaException(f"Error in DataIngestionConfig initialization: {e}") from e

@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation phase."""
    data_validation_dir: Path = field(
        default_factory=lambda: training_pipeline_config.artifact_dir / DATA_VALIDATION_DIR_NAME
    )
    drift_report_file_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize drift report path and create directories."""
        try:
            object.__setattr__(self, 'drift_report_file_path',
                             self.data_validation_dir / DATA_VALIDATION_DRIFT_REPORT_DIR / 
                             DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
            
            self.data_validation_dir.mkdir(parents=True, exist_ok=True)
            self.drift_report_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Data validation directories created successfully")
        except Exception as e:
            raise USvisaException(f"Error in DataValidationConfig initialization: {e}") from e

@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for data transformation phase."""
    data_transformation_dir: Path = field(
        default_factory=lambda: training_pipeline_config.artifact_dir / DATA_TRANSFORMATION_DIR_NAME
    )
    transformed_train_file_path: Path = field(init=False)
    transformed_test_file_path: Path = field(init=False)
    transformed_object_file_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize transformation paths and create directories."""
        try:
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
            self.data_transformation_dir.mkdir(parents=True, exist_ok=True)
            self.transformed_train_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.transformed_object_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Data transformation directories created successfully")
        except Exception as e:
            raise USvisaException(f"Error in DataTransformationConfig initialization: {e}") from e

@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training phase."""
    model_trainer_dir: Path = field(
        default_factory=lambda: training_pipeline_config.artifact_dir / MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: Path = field(init=False)
    expected_accuracy: float = field(default=MODEL_TRAINER_EXPECTED_SCORE)
    model_config_file_path: Path = field(default=Path(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH))

    def __post_init__(self):
        """Initialize model paths and create directories."""
        try:
            object.__setattr__(self, 'trained_model_file_path',
                             self.model_trainer_dir / MODEL_TRAINER_TRAINED_MODEL_DIR / MODEL_FILE_NAME)
            
            self.model_trainer_dir.mkdir(parents=True, exist_ok=True)
            self.trained_model_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Model trainer directories created successfully")
        except Exception as e:
            raise USvisaException(f"Error in ModelTrainerConfig initialization: {e}") from e

@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation phase."""
    changed_threshold_score: float = field(default=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE)
    bucket_name: str = field(default=MODEL_BUCKET_NAME)
    s3_model_key_path: str = field(default=MODEL_FILE_NAME)

@dataclass(frozen=True)
class ModelPusherConfig:
    """Configuration for model pushing phase."""
    bucket_name: str = field(default=MODEL_BUCKET_NAME)
    s3_model_key_path: str = field(default=MODEL_FILE_NAME)

@dataclass(frozen=True)
class USvisaPredictorConfig:
    """Configuration for US visa prediction service."""
    model_file_path: str = field(default=MODEL_FILE_NAME)
    model_bucket_name: str = field(default=MODEL_BUCKET_NAME)

# Initialize the training pipeline configuration
try:
    training_pipeline_config = TrainingPipelineConfig()
    logging.info("Training pipeline configuration initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize training pipeline configuration: {e}")
    raise