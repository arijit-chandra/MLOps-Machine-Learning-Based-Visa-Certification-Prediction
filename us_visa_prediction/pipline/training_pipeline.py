from typing import Optional
import sys
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.components.data_ingestion import DataIngestion
from us_visa_prediction.components.data_validation import DataValidation
from us_visa_prediction.components.data_transformation import DataTransformation
from us_visa_prediction.components.model_trainer import ModelTrainer
from us_visa_prediction.components.model_evaluation import ModelEvaluation
# from us_visa_prediction.components.model_pusher import ModelPusher

from us_visa_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    # ModelPusherConfig
)

from us_visa_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

class TrainPipeline:
    """
    TrainPipeline orchestrates the entire machine learning pipeline from data ingestion to model deployment.
    """
    
    def __init__(self):
        """Initialize pipeline configurations"""
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig()
            # self.model_pusher_config = ModelPusherConfig()
            
            logging.info("Training pipeline configurations initialized successfully")
        except Exception as e:
            raise USvisaException(f"Error initializing training pipeline: {e}", sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiate the data ingestion process.
        
        Returns:
            DataIngestionArtifact: Artifact containing paths to ingested data
            
        Raises:
            USvisaException: If data ingestion fails
        """
        try:
            logging.info("Starting data ingestion phase")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException("Failed to ingest data", sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Initiate the data validation process.
        
        Args:
            data_ingestion_artifact: Artifact from data ingestion phase
            
        Returns:
            DataValidationArtifact: Artifact containing validation results
            
        Raises:
            USvisaException: If data validation fails
        """
        try:
            logging.info("Starting data validation phase")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException("Failed to validate data", sys) from e

    def start_data_transformation(
        self, 
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Initiate the data transformation process.
        
        Args:
            data_ingestion_artifact: Artifact from data ingestion phase
            data_validation_artifact: Artifact from data validation phase
            
        Returns:
            DataTransformationArtifact: Artifact containing transformed data
            
        Raises:
            USvisaException: If data transformation fails
        """
        try:
            logging.info("Starting data transformation phase")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise USvisaException("Failed to transform data", sys) from e

    def start_model_trainer(
        self, 
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        Initiate the model training process.
        
        Args:
            data_transformation_artifact: Artifact from data transformation phase
            
        Returns:
            ModelTrainerArtifact: Artifact containing trained model
            
        Raises:
            USvisaException: If model training fails
        """
        try:
            logging.info("Starting model training phase")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise USvisaException("Failed to train model", sys) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        """
        Initiate the model evaluation process.
        
        Args:
            data_ingestion_artifact: Artifact from data ingestion phase
            model_trainer_artifact: Artifact from model training phase
            
        Returns:
            ModelEvaluationArtifact: Artifact containing evaluation results
            
        Raises:
            USvisaException: If model evaluation fails
        """
        try:
            logging.info("Starting model evaluation phase")
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException("Failed to evaluate model", sys) from e

    # def start_model_pusher(
    #     self,
    #     model_evaluation_artifact: ModelEvaluationArtifact
    # ) -> Optional[ModelPusherArtifact]:
    #     """
    #     Initiate the model pushing process if evaluation criteria are met.
        
    #     Args:
    #         model_evaluation_artifact: Artifact from model evaluation phase
            
    #     Returns:
    #         Optional[ModelPusherArtifact]: Artifact containing pushed model info if successful
            
    #     Raises:
    #         USvisaException: If model pushing fails
    #     """
    #     try:
    #         if not model_evaluation_artifact.is_model_accepted:
    #             logging.info("Model not accepted for deployment")
    #             return None
                
    #         logging.info("Starting model pusher phase")
    #         model_pusher = ModelPusher(
    #             model_evaluation_artifact=model_evaluation_artifact,
    #             model_pusher_config=self.model_pusher_config
    #         )
    #         model_pusher_artifact = model_pusher.initiate_model_pusher()
    #         logging.info(f"Model pushing completed. Artifact: {model_pusher_artifact}")
    #         return model_pusher_artifact
    #     except Exception as e:
    #         raise USvisaException("Failed to push model", sys) from e

    def run_pipeline(self) -> Optional[ModelPusherArtifact]:
        """
        Execute the complete training pipeline from data ingestion to model pushing.
        
        Returns:
            Optional[ModelPusherArtifact]: Artifact containing pushed model info if successful
            
        Raises:
            USvisaException: If any pipeline stage fails
        """
        try:
            logging.info("Starting training pipeline execution")
            
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            
            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            
            # Model Training
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            
            # # Model Pushing (if accepted)
            # model_pusher_artifact = self.start_model_pusher(
            #     model_evaluation_artifact=model_evaluation_artifact
            # )
            
            logging.info("Training pipeline execution completed successfully")
            # return model_pusher_artifact
            
        except Exception as e:
            logging.error("Pipeline execution failed")
            raise USvisaException("Pipeline execution failed", sys) from e