### this is azure_storage.py
import sys
from typing import Union, List
from io import StringIO
import pandas as pd

from us_visa_prediction.configuration.azure_connection import AzureClient
from us_visa_prediction.constants import MODEL_CONTAINER_NAME
from us_visa_prediction.exception import USVisaPredictionException
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
            self.container_name = container_name
            self.container_client = self.azure_client.blob_service_client.get_container_client(container_name)
        except Exception as e:
            logging.error(f"Failed to initialize Azure Storage Service: {e}")
            raise USVisaPredictionException(e, sys)

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
            raise USVisaPredictionException(e, sys)

    def upload_blob(self, blob_name: str, data: Union[str, bytes, pd.DataFrame]):
        """
        Upload blob to the container
        
        Args:
            blob_name (str): Name of the blob
            data (Union[str, bytes, pd.DataFrame]): Data to upload
        """
        try:
            # If DataFrame is passed, convert to CSV string
            if isinstance(data, pd.DataFrame):
                data_str = data.to_csv(index=False)
            elif isinstance(data, (str, bytes)):
                data_str = data
            else:
                raise ValueError("Unsupported data type for upload")

            self.container_client.upload_blob(name=blob_name, data=data_str, overwrite=True)
            logging.info(f"Successfully uploaded blob: {blob_name}")
        except Exception as e:
            logging.error(f"Blob upload failed for {blob_name}: {e}")
            raise USVisaPredictionException(e, sys)

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
            raise USVisaPredictionException(e, sys)

    def download_csv(self, blob_name: str) -> pd.DataFrame:
        """
        Download CSV blob and return as pandas DataFrame
        
        Args:
            blob_name (str): Name of the CSV blob
        
        Returns:
            pd.DataFrame: Downloaded data as DataFrame
        """
        try:
            csv_content = self.download_blob(blob_name)
            return pd.read_csv(StringIO(csv_content))
        except Exception as e:
            logging.error(f"CSV download failed for {blob_name}: {e}")
            raise USVisaPredictionException(e, sys)

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
            raise USVisaPredictionException(e, sys)
