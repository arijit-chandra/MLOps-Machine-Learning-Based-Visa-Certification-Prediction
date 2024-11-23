from us_visa_prediction.entity.config_entity import ModelEvaluationConfig
from us_visa_prediction.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa_prediction.logger import logging
from us_visa_prediction.cloud_storage.azure_storage import AzureStorageService
import sys
import numpy as np
import pandas as pd
from typing import Optional
from us_visa_prediction.entity.cloud_estimator import USvisaEstimator  
from dataclasses import dataclass
from us_visa_prediction.entity.estimator import USvisaModel
from us_visa_prediction.entity.estimator import TargetValueMapping

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.azure_storage = AzureStorageService(
                container_name=model_eval_config.container_name
            )
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering steps including log transforms and percentiles"""
        try:
            # Add company age
            df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
            
            # Log transformations
            df['log_company_age'] = np.log1p(df['company_age'])
            df['log_no_of_employees'] = np.log1p(df['no_of_employees'])
            
            # Wage percentiles by continent
            df['wage_percentile'] = df.groupby('continent')['prevailing_wage'].rank(pct=True)
            
            logging.info("Added engineered features: log transforms and wage percentiles")
            return df
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_best_model(self) -> Optional[USvisaEstimator]:
        try:
            container_name = self.model_eval_config.container_name  
            model_path = self.model_eval_config.model_path
            
            if not self.azure_storage.is_blob_exists(model_path):
                return None
            
            usvisa_estimator = USvisaEstimator(
                container_name=container_name,
                model_path=model_path
            )
            return usvisa_estimator
        except Exception as e:
            logging.error(f"Error retrieving best model: {e}")
            return None

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            # Load and prepare test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Apply feature engineering
            test_df = self._perform_feature_engineering(test_df)

            # Separate features and target
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            
            # Convert target values using mapping
            y = y.replace(TargetValueMapping()._asdict())

            # Load and evaluate trained model
            trained_model = USvisaModel(
                model_file_path=self.model_trainer_artifact.trained_model_file_path
            )
            y_hat_trained = trained_model.predict(x)
            trained_model_f1_score = f1_score(y, y_hat_trained)

            # Evaluate production model if available
            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            # Compare models
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            
            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()
            azure_model_path = self.model_eval_config.model_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                azure_model_path=azure_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise USvisaException(e, sys) from e