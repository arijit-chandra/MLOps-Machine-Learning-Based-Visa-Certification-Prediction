from dataclasses import dataclass
from typing import Dict
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging
from us_visa_prediction.utils.main_utils import read_yaml_file


@dataclass
class ModelDetails:
    """Contains details about the best model found during model selection"""
    best_model: object
    best_score: float
    best_parameters: Dict


class ModelFactory:
    """Factory class for creating and evaluating machine learning models based on configuration"""
    
    def __init__(self, model_config_path: str):
        """
        Initialize ModelFactory with path to model config yaml file
        
        Args:
            model_config_path: Path to the model configuration YAML file
        """
        try:
            self.config = read_yaml_file(model_config_path)
            self.grid_search_cv_module = GridSearchCV
            self.model_dict = self._get_model_dict()
            
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _get_model_dict(self) -> Dict:
        """
        Load model details from config file
        
        Returns:
            Dictionary mapping model names to their configurations
        """
        try:
            model_details = self.config['model_selection']
            model_dict = {}
            
            for model_name, model_info in model_details.items():
                module_name = model_info['module']
                class_name = model_info['class']
                grid_search_params = model_info.get('search_param_grid', {})
                
                model_class = self._load_model_class(module_name, class_name)
                model_dict[model_name] = {
                    'model': model_class(),
                    'params': grid_search_params
                }
                
            return model_dict
            
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _load_model_class(self, module_name: str, class_name: str) -> object:
        """
        Dynamically load model class from module
        
        Args:
            module_name: Name of the module containing the model class
            class_name: Name of the model class
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model is not supported
        """
        try:
            model_map = {
                'sklearn.ensemble.RandomForestClassifier': RandomForestClassifier,
                'sklearn.ensemble.GradientBoostingClassifier': GradientBoostingClassifier,
                'sklearn.linear_model.LogisticRegression': LogisticRegression,
                'sklearn.neighbors.KNeighborsClassifier': KNeighborsClassifier,
                'sklearn.svm.SVC': SVC,
                'sklearn.tree.DecisionTreeClassifier': DecisionTreeClassifier,
                'sklearn.naive_bayes.GaussianNB': GaussianNB,
                'xgboost.XGBClassifier': XGBClassifier
            }
            
            model_path = f"{module_name}.{class_name}"
            if model_path in model_map:
                return model_map[model_path]
            else:
                supported_models = list(model_map.keys())
                raise ValueError(
                    f"Model {model_path} not supported. Supported models are: {supported_models}"
                )
                
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_best_model(self, X: np.ndarray, y: np.ndarray, base_accuracy: float = 0.6) -> ModelDetails:
        """
        Evaluate all models from config and return the best one
        
        Args:
            X: Training features
            y: Training labels
            base_accuracy: Minimum accuracy threshold for model selection
            
        Returns:
            ModelDetails containing the best performing model
            
        Raises:
            Exception: If no model achieves the base accuracy
        """
        try:
            logging.info("Starting model evaluation")
            best_score = 0.0
            best_model = None
            best_params = None
            
            for model_name, model_info in self.model_dict.items():
                logging.info(f"Evaluating model: {model_name}")
                
                try:
                    grid_search = self.grid_search_cv_module(
                        estimator=model_info['model'],
                        param_grid=model_info['params'],
                        scoring='f1',
                        n_jobs=-1,
                        cv=5
                    )
                    
                    grid_search.fit(X, y)
                    
                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        logging.info(f"Found new best model: {model_name} with score: {best_score:.3f}")
                        
                except Exception as e:
                    logging.warning(f"Error evaluating model {model_name}: {str(e)}")
                    continue
                    
            if best_model is None:
                raise Exception("No models were successfully trained")
                    
            if best_score < base_accuracy:
                logging.info(f"No model achieved base accuracy of {base_accuracy}")
                raise Exception(f"No model achieved base accuracy of {base_accuracy}")
                
            logging.info(f"Best model found with score: {best_score:.3f}")
            return ModelDetails(
                best_model=best_model,
                best_score=best_score,
                best_parameters=best_params
            )
            
        except Exception as e:
            raise USvisaException(e, sys) from e