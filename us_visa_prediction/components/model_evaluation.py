import sys
import pickle
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from us_visa_prediction.entity.config_entity import ModelEvaluationConfig
from us_visa_prediction.cloud_storage.azure_storage import AzureStorageService
from us_visa_prediction.entity.artifact_entity import (
    DataIngestionArtifact, 
    ModelTrainerArtifact, 
    ModelEvaluationArtifact
)
from us_visa_prediction.constants import CURRENT_YEAR, TARGET_COLUMN
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.entity.estimator import TargetValueMapping
from us_visa_prediction.utils.main_utils import save_object, load_object

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(
        self, 
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        """
        Initialize Model Evaluation
        
        Args:
            model_eval_config (ModelEvaluationConfig): Model evaluation configuration
            data_ingestion_artifact (DataIngestionArtifact): Data ingestion artifact
            model_trainer_artifact (ModelTrainerArtifact): Model trainer artifact
        """
        try:
            logging.info(f"{'=' * 20} Model Evaluation {'=' * 20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            logging.info(f"Initialize Model Evaluation container_name: {self.model_eval_config.container_name}")
            # Azure storage service for model management
            self.azure_storage_service = AzureStorageService(
                container_name=self.model_eval_config.container_name
            )
        except Exception as e:
            raise USvisaException(e, sys)

    def get_best_model(self):
        """
        Get the best model from Azure storage
        
        Returns:
            Loaded model or None if no model exists
        """
        try:
            blob_name = self.model_eval_config.blob_model_key_path
            if self.azure_storage_service.is_blob_exists(blob_name):
                # Download blob content as bytes
                existing_model_blob = self.azure_storage_service.download_blob(blob_name)
                
                # Create a temporary file path for the model
                existing_model_path = str(self.model_eval_config.model_path)
                
                # Save the downloaded bytes to a file
                with open(existing_model_path, 'wb') as f:
                    f.write(existing_model_blob)
                
                # Load the model using pickle
                with open(existing_model_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logging.error(f"Error getting best model: {e}")
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Evaluate and compare models
        
        Returns:
            ModelEvaluationArtifact: Result of model evaluation
        """
        try:
            # Load test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Perform feature engineering
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            
            # Log transformations
            test_df['log_company_age'] = np.log1p(test_df['company_age'])
            test_df['log_no_of_employees'] = np.log1p(test_df['no_of_employees'])
            
            # Wage percentiles by continent
            test_df['wage_percentile'] = test_df.groupby('continent')['prevailing_wage'].rank(pct=True)

            # Prepare features and target
            x_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN].replace(
                TargetValueMapping()._asdict()
            )

            # Evaluate trained model
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            # Get best existing model
            best_model = self.get_best_model()
            best_model_f1_score = None

            # Evaluate best model if exists
            if best_model is not None:
                y_pred_best_model = best_model.predict(x_test)
                best_model_f1_score = f1_score(y_test, y_pred_best_model, average='weighted')

            # Determine model acceptance
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = trained_model_f1_score > tmp_best_model_score
            changed_accuracy = trained_model_f1_score - tmp_best_model_score

            # Create model evaluation artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                changed_accuracy=changed_accuracy,
                blob_model_path=self.model_eval_config.blob_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )

            # Push model to storage if accepted
            if is_model_accepted:
                blob_name = self.model_eval_config.blob_model_key_path
                
                # Use pickle to serialize the model
                with open(self.model_trainer_artifact.trained_model_file_path, 'rb') as model_file:
                    model_bytes = model_file.read()
                
                # Upload the serialized model
                self.azure_storage_service.upload_blob(
                    blob_name=blob_name,
                    data=model_bytes
                )

            logging.info(f"Model Evaluation Result: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            logging.error(f"Model evaluation failed: {str(e)}")
            raise USvisaException(f"Failed to evaluate model: {str(e)}", sys) from e