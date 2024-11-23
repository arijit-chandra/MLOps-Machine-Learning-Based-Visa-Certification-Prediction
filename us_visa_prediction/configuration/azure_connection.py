### this is azure_conncection.py
from azure.storage.blob import BlobServiceClient
from us_visa_prediction.constants import AZURE_STORAGE_CONNECTION_STRING
from us_visa_prediction.logger import logging
from us_visa_prediction.exception import USVisaPredictionException
import sys

class AzureClient:
    blob_service_client = None

    def __init__(self):
        """
        Initialize Azure Blob Storage client using constants
        """
        if AzureClient.blob_service_client is None:
            try:
                AzureClient.blob_service_client = BlobServiceClient.from_connection_string(
                    conn_str=AZURE_STORAGE_CONNECTION_STRING
                )
                logging.info("Azure Blob Service Client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Azure Blob Service Client: {e}")
                raise
        
        self.blob_service_client = AzureClient.blob_service_client

    def download_file(self, container_name: str, blob_name: str, local_file_path: str):
        """
        Download a file from Azure Blob Storage
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            with open(local_file_path, "wb") as file:
                blob_data = blob_client.download_blob()
                file.write(blob_data.readall())
            
            return local_file_path
        except Exception as e:
            raise USVisaPredictionException(e, sys)

    def upload_file(self, local_file_path: str, container_name: str, blob_name: str):
        """
        Upload a file to Azure Blob Storage
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            with open(local_file_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        except Exception as e:
            raise USVisaPredictionException(e, sys)

    def is_blob_exists(self, container_name: str, blob_name: str) -> bool:
        """
        Check if a blob exists in the container
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            return container_client.get_blob_client(blob_name).exists()
        except Exception as e:
            raise USVisaPredictionException(e, sys)