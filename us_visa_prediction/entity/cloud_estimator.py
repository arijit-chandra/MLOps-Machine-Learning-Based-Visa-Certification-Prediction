### This is cloud_estimator.py file
from us_visa_prediction.cloud_storage.azure_storage import AzureStorageService
import os
import joblib

class USvisaEstimator:
    """
    Cloud-based model estimator for US Visa Prediction
    """
    def __init__(self, container_name: str, model_path: str):
        """
        Initialize with Azure storage configuration
       
        Args:
            container_name (str): Name of the Azure container
            model_path (str): Path to the model file in the container
        """
        self.container_name = container_name
        self.model_path = model_path
        self.azure_storage = AzureStorageService(container_name=container_name)
   
    def is_model_present(self) -> bool:
        """
        Check if model exists in Azure storage
       
        Returns:
            bool: True if model exists, False otherwise
        """
        try:
            return self.azure_storage.is_blob_exists(self.model_path)
        except Exception:
            return False
       
    def load_model(self) -> object:
        """
        Load model from Azure storage
       
        Returns:
            object: Loaded machine learning model
        """
        temp_file = "temp_model.pkl"
        try:
            # Download blob content directly
            model_content = self.azure_storage.download_blob(self.model_path)
            
            # Save to temp file
            with open(temp_file, 'wb') as f:
                f.write(model_content.encode('latin1'))
           
            # Load model
            model = joblib.load(temp_file)
            
            # Cleanup
            os.remove(temp_file)
            return model
        except Exception as e:
            # Ensure temp file is removed even if an error occurs
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise Exception(f"Failed to load model: {str(e)}")
   
    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save a local model file to Azure storage
       
        Args:
            from_file (str): Local path to the model file to upload
            remove (bool, optional): If True, removes the local file after upload
        """
        if not os.path.exists(from_file):
            raise FileNotFoundError(f"Model file not found at: {from_file}")
       
        try:
            # Read file content
            with open(from_file, 'rb') as data:
                model_bytes = data.read()
            
            # Upload to Azure storage
            self.azure_storage.upload_blob(self.model_path, model_bytes)
           
            # Remove local file if specified
            if remove:
                os.remove(from_file)
               
        except Exception as e:
            raise Exception(f"Failed to save model to Azure storage: {str(e)}")
   
    def predict(self, X):
        """
        Make predictions using the cloud-stored model
       
        Args:
            X: Input data for prediction
       
        Returns:
            Predictions from the loaded model
        """
        try:
            model = self.load_model()
            return model.predict(X)
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")