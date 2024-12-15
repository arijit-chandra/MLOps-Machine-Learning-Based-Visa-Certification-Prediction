import os
import sys
from us_visa_prediction.cloud_storage.azure_storage import AzureStorageService
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa_prediction.entity.config_entity import ModelPusherConfig
from us_visa_prediction.entity.cloud_estimator import USvisaEstimator

class ModelPusher:
    def __init__(self, 
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initialize ModelPusher with evaluation artifact and pusher configuration
       
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        try:
            self.azure_storage = AzureStorageService(
                container_name=model_pusher_config.container_name
            )
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
            self.usvisa_estimator = USvisaEstimator(
                container_name=model_pusher_config.container_name,
                model_path=model_pusher_config.blob_model_path  
            )
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _validate_model_file(self, model_path: str) -> bool:
        """
        Validate the model file before upload
        
        :param model_path: Path to the model file
        :return: Boolean indicating if the model file is valid
        """
        try:
            # Check file existence
            if not os.path.exists(model_path):
                logging.error(f"Model file does not exist: {model_path}")
                return False
            
            # Check file size (optional: adjust max size as needed)
            max_model_size_mb = 500  # 500 MB max file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            if file_size_mb > max_model_size_mb:
                logging.error(f"Model file too large: {file_size_mb:.2f} MB (max {max_model_size_mb} MB)")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating model file: {e}")
            return False

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiate model pushing to Azure blob storage
       
        :return: Model pusher artifact
        :raises USvisaException: If model pushing fails
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            # Validate model file path from evaluation artifact
            model_path = self.model_evaluation_artifact.trained_model_path
            
            # Validate the model file
            if not self._validate_model_file(model_path):
                raise ValueError(f"Invalid model file: {model_path}")
            
            logging.info(f"Preparing to upload model from: {model_path}")
            
            # Read and upload model file
            with open(model_path, 'rb') as model_file:
                self.azure_storage.upload_blob(
                    blob_name=self.model_pusher_config.blob_model_path,  
                    data=model_file.read()
                )
        
            # Create model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(
                container_name=self.model_pusher_config.container_name,
                blob_model_path=self.model_pusher_config.blob_model_path
            )
        
            logging.info("Model successfully uploaded to Azure blob storage")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
        
            return model_pusher_artifact
    
        except Exception as e:
            logging.error(f"Model pushing failed: {e}")
            raise USvisaException(e, sys) from e