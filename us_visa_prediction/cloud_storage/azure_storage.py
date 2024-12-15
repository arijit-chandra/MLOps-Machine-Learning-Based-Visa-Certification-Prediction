### this is azure_storage.py
import sys
from typing import Union, List
from io import StringIO
import pandas as pd
from typing import Any, Union

from us_visa_prediction.configuration.azure_connection import AzureClient
from us_visa_prediction.constants import MODEL_CONTAINER_NAME
from us_visa_prediction.exception import USvisaException
from us_visa_prediction.logger import logging

class AzureStorageService:
    def __init__(self, container_name: str = MODEL_CONTAINER_NAME):
        """
        Initialize Azure Storage Service
        
        Args:
            container_name (str, optional): Name of the container to work with. 
                                            Defaults to MODEL_CONTAINER_NAME.
        """
        try:
            self.azure_client = AzureClient()
            logging.info(f"Container name: {container_name}")
            self.container_name = container_name
            self.container_client = self.azure_client.blob_service_client.get_container_client(container_name)
        except Exception as e:
            logging.error(f"Failed to initialize Azure Storage Service: {e}")
            raise USvisaException(e, sys)

    def list_blobs(self) -> List[str]:
        """
        List all blob names in the container
        
        Returns:
            List[str]: List of blob names
        """
        try:
            return [blob.name for blob in self.container_client.list_blobs()]
        except Exception as e:
            logging.error(f"Failed to list blobs: {e}")
            raise USvisaException(e, sys)

    def upload_blob(self, blob_name: str, data: Union[str, bytes, pd.DataFrame, Any]):
        """
        Upload blob to the container
        
        Args:
            blob_name (str): Name of the blob
            data (Union[str, bytes, pd.DataFrame, Any]): Data to upload
        """
        try:
            # If DataFrame is passed, convert to CSV string
            if isinstance(data, pd.DataFrame):
                data_str = data.to_csv(index=False)
                self.container_client.upload_blob(name=blob_name, data=data_str, overwrite=True)
            # If bytes or binary file content is passed
            elif isinstance(data, bytes):
                self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            elif isinstance(data, (str, pd.Series)):
                data_str = str(data)
                self.container_client.upload_blob(name=blob_name, data=data_str, overwrite=True)
            else:
                raise ValueError(f"Unsupported data type for upload: {type(data)}")

            logging.info(f"Successfully uploaded blob: {blob_name}")
        except Exception as e:
            logging.error(f"Blob upload failed for {blob_name}: {e}")
            raise USvisaException(e, sys)

    def download_blob(self, blob_name: str) -> str:
        """
        Download blob content as string
        
        Args:
            blob_name (str): Name of the blob to download
        
        Returns:
            str: Blob content as string
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall().decode('utf-8')
        except Exception as e:
            logging.error(f"Blob download failed for {blob_name}: {e}")
            raise USvisaException(e, sys)

    def download_blob(self, blob_name: str) -> bytes:
        """
        Download blob content as bytes
        
        Args:
            blob_name (str): Name of the blob to download
        
        Returns:
            bytes: Blob content as bytes
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            logging.error(f"Blob download failed for {blob_name}: {e}")
            raise USvisaException(e, sys)                                                                           

    def is_blob_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists in the container
        
        Args:
            blob_name (str): Name of the blob to check
        
        Returns:
            bool: True if blob exists, False otherwise
        """
        try:
            return self.container_client.get_blob_client(blob_name).exists()
        except Exception as e:
            logging.error(f"Failed to check blob existence for {blob_name}: {e}")
            raise USvisaException(e, sys)
        
    def download_pickle(self, blob_name: str):
        """
        Download and load a pickle file
        
        Args:
            blob_name (str): Name of the pickle blob to download
        
        Returns:
            Any: Deserialized pickle object
        """
        try:
            import pickle
            blob_content = self.download_blob(blob_name)
            return pickle.loads(blob_content)
        except Exception as e:
            logging.error(f"Pickle download failed for {blob_name}: {e}")
            raise USvisaException(e, sys)
