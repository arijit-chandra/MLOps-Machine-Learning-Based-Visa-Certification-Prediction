### artifact_entity.py file
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from us_visa_prediction.logger import logging
from us_visa_prediction.exception import USvisaException

@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact class for storing data ingestion results.
    Contains paths to trained and test datasets.
    """
    trained_file_path: Path
    test_file_path: Path

    def __post_init__(self):
        try:
            # Validate file paths exist
            if not self.trained_file_path.exists():
                raise USvisaException(f"Training file not found: {self.trained_file_path}")
            if not self.test_file_path.exists():
                raise USvisaException(f"Test file not found: {self.test_file_path}")
            
            logging.info(f"Data ingestion artifact validated: {self}")
        except Exception as e:
            logging.error(f"Error in DataIngestionArtifact validation: {e}")
            raise USvisaException(f"Error in DataIngestionArtifact validation: {e}") from e

@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact class for storing data validation results.
    Contains validation status, message, and drift report path.
    """
    validation_status: bool
    message: str
    drift_report_file_path: Optional[Path] = None

    def __post_init__(self):
        try:
            # Validate drift report path exists if validation was successful
            if self.validation_status and self.drift_report_file_path:
                if not self.drift_report_file_path.exists():
                    raise USvisaException(f"Drift report not found: {self.drift_report_file_path}")
            
            logging.info(f"Data validation artifact validated: {self}")
        except Exception as e:
            logging.error(f"Error in DataValidationArtifact validation: {e}")
            raise USvisaException(f"Error in DataValidationArtifact validation: {e}") from e

@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact class for storing data transformation results.
    Contains paths to transformed data and preprocessing object.
    """
    transformed_object_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path

    def __post_init__(self):
        try:
            # Validate all paths exist
            paths_to_verify = [
                self.transformed_object_file_path,
                self.transformed_train_file_path,
                self.transformed_test_file_path
            ]
            
            for path in paths_to_verify:
                if not path.exists():
                    raise USvisaException(f"Transformed file not found: {path}")
            
            logging.info(f"Data transformation artifact validated: {self}")
        except Exception as e:
            logging.error(f"Error in DataTransformationArtifact validation: {e}")
            raise USvisaException(f"Error in DataTransformationArtifact validation: {e}") from e

@dataclass(frozen=True)
class ClassificationMetricArtifact:
    """
    Artifact class for storing classification metrics.
    Contains various performance metrics.
    """
    f1_score: float
    precision_score: float
    recall_score: float

    def __post_init__(self):
        try:
            # Validate metric values are in valid range [0, 1]
            metrics = {
                'f1_score': self.f1_score,
                'precision_score': self.precision_score,
                'recall_score': self.recall_score
            }
            
            for metric_name, value in metrics.items():
                if not 0 <= value <= 1:
                    raise USvisaException(
                        f"Invalid {metric_name}: {value}. Metrics must be between 0 and 1"
                    )
            
            logging.info(f"Classification metrics validated: {self}")
        except Exception as e:
            logging.error(f"Error in ClassificationMetricArtifact validation: {e}")
            raise USvisaException(f"Error in ClassificationMetricArtifact validation: {e}") from e

    def to_dict(self) -> dict:
        """Convert metrics to dictionary format."""
        return {
            'f1_score': self.f1_score,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score
        }

@dataclass(frozen=True)
class ModelTrainerArtifact:
    """
    Artifact class for storing model training results.
    Contains model path and performance metrics.
    """
    trained_model_file_path: Path
    metric_artifact: ClassificationMetricArtifact

    def __post_init__(self):
        try:
            # Validate model file exists
            if not self.trained_model_file_path.exists():
                raise USvisaException(f"Trained model not found: {self.trained_model_file_path}")
            
            logging.info(f"Model trainer artifact validated: {self}")
        except Exception as e:
            logging.error(f"Error in ModelTrainerArtifact validation: {e}")
            raise USvisaException(f"Error in ModelTrainerArtifact validation: {e}") from e

@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """
    Artifact class for storing model evaluation results.
    Contains evaluation metrics and model paths.
    """
    is_model_accepted: bool
    changed_accuracy: float
    blob_model_path: str
    trained_model_path: Path
    
    def __post_init__(self):
        try:
            # Validate accuracy change is in reasonable range
            if not -1 <= self.changed_accuracy <= 1:
                raise USvisaException(
                    f"Invalid accuracy change: {self.changed_accuracy}. "
                    "Should be between -1 and 1"
                )
            
            # Validate trained model path if model is accepted
            if self.is_model_accepted and not self.trained_model_path.exists():
                raise USvisaException(f"Accepted model not found: {self.trained_model_path}")
            
            logging.info(f"Model evaluation artifact validated: {self}")
        except Exception as e:
            logging.error(f"Error in ModelEvaluationArtifact validation: {e}")
            raise USvisaException(f"Error in ModelEvaluationArtifact validation: {e}") from e

@dataclass(frozen=True)
class ModelPusherArtifact:
    """
    Artifact class for storing model deployment information.
    Contains Azure Blob Storage container details and model path.
    """
    container_name: str
    blob_model_path: str

    def __post_init__(self):
        try:
            # Basic validation of container name and path
            if not self.container_name or not self.container_name.strip():
                raise USvisaException("Container name cannot be empty")
            if not self.blob_model_path or not self.blob_model_path.strip():
                raise USvisaException("Blob model path cannot be empty")
            
            logging.info(f"Model pusher artifact validated: {self}")
        except Exception as e:
            logging.error(f"Error in ModelPusherArtifact validation: {e}")
            raise USvisaException(f"Error in ModelPusherArtifact validation: {e}") from e

    def get_blob_uri(self) -> str:
        """Generate full Azure Blob Storage URI for the model."""
        return f"https://{self.container_name}.blob.core.windows.net/{self.blob_model_path}"
