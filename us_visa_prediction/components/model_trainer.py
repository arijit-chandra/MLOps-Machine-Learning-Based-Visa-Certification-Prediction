import sys
from typing import Tuple
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa_prediction.entity.config_entity import ModelTrainerConfig
from us_visa_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from us_visa_prediction.entity.estimator import USvisaModel
from us_visa_prediction.components.model_factory import ModelFactory, ModelDetails


class ModelTrainer:
    """
    ModelTrainer is responsible for training and evaluating machine learning models
    using the transformed data from the DataTransformation stage.
    """
    
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """
        Initialize ModelTrainer with artifacts and configuration
        
        Args:
            data_transformation_artifact: Output reference of data transformation artifact stage
            model_trainer_config: Configuration for model training
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_model_object_and_report(
        self,
        train: np.array,
        test: np.array
    ) -> Tuple[ModelDetails, ClassificationMetricArtifact]:
        """
        Perform model selection and evaluation
        
        Args:
            train: Training data array containing features and target
            test: Testing data array containing features and target
            
        Returns:
            Tuple containing:
                - ModelDetails with best model and its parameters
                - ClassificationMetricArtifact with model performance metrics
                
        Raises:
            USvisaException: If model training or evaluation fails
        """
        try:
            logging.info("Splitting train and test input data")
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            logging.info("Using ModelFactory to get best model object and report")
            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_config_file_path
            )
            
            best_model_detail = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy
            )
            
            logging.info("Making predictions on test set using best model")
            y_pred = best_model_detail.best_model.predict(x_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            logging.info(f"Model Metrics - F1: {f1:.3f}, Precision: {precision:.3f}, "
                        f"Recall: {recall:.3f}, Accuracy: {accuracy:.3f}")
            
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate the model training process
        
        This function:
            1. Loads transformed training and testing data
            2. Gets the best model through model selection
            3. Creates a USvisaModel with preprocessing and model objects
            4. Saves the model and returns training artifacts
            
        Returns:
            ModelTrainerArtifact containing paths to saved models and metric results
            
        Raises:
            USvisaException: If model training process fails
            Exception: If no model meets the base accuracy threshold
        """
        try:
            logging.info("Loading transformed training and testing arrays")
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            
            logging.info("Getting best model and performance metrics")
            best_model_detail, metric_artifact = self.get_model_object_and_report(
                train=train_arr,
                test=test_arr
            )
            
            logging.info("Loading preprocessing object")
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.error("No best model found with score more than base accuracy")
                raise Exception("No best model found with score more than base accuracy")

            logging.info("Creating USvisaModel with preprocessor and trained model")
            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )
            
            logging.info("Saving trained model object")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=usvisa_model
            )

            logging.info("Creating model trainer artifact")
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise USvisaException(e, sys) from e