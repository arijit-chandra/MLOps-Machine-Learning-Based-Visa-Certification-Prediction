import sys
from us_visa_prediction.cloud_storage.azure_storage import AzureStorageService
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa_prediction.entity.config_entity import ModelPusherConfig
from us_visa_prediction.entity.cloud_estimator import USvisaEstimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initialize ModelPusher with evaluation artifact and pusher configuration
        
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        try:
            self.azure_storage = AzureStorageService(container_name=model_pusher_config.container_name)
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
            self.usvisa_estimator = USvisaEstimator(
                container_name=model_pusher_config.container_name,
                model_path=model_pusher_config.azure_model_key_path
            )
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiate model pushing to Azure blob storage
        
        :return: Model pusher artifact
        :raises USvisaException: If model pushing fails
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading trained model to Azure blob storage")
            
            # Read model file and upload
            with open(self.model_evaluation_artifact.trained_model_path, 'rb') as model_file:
                self.azure_storage.upload_blob(
                    blob_name=self.model_pusher_config.azure_model_key_path, 
                    data=model_file.read()
                )
            
            # Create model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(
                container_name=self.model_pusher_config.container_name,
                azure_model_path=self.model_pusher_config.azure_model_key_path
            )
            
            logging.info("Model successfully uploaded to Azure blob storage")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            
            return model_pusher_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e