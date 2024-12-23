from azure.storage.blob import BlobServiceClient
import os

from us_visa_prediction.constants import AZURE_ACCOUNT_URL
from us_visa_prediction.logger import logging
from us_visa_prediction.exception import USvisaException
import sys

class AzureClient:
    def __init__(self):
        """
        Initialize Azure Blob Storage client using Azure AD credentials
        """
        try:
            # Create blob service client
            account_url=os.getenv(AZURE_ACCOUNT_URL)
            self.blob_service_client = BlobServiceClient.from_connection_string(account_url)
            
            logging.info("Azure Blob Service Client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Azure Blob Service Client: {e}")
            raise USvisaException(e, sys)

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
            raise USvisaException(e, sys)

    def download_blob_data(self, container_name: str, blob_name: str) -> str:
        """
        Download blob data as a string
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            data = blob_client.download_blob().readall().decode("utf-8")
            return data
        except Exception as e:
            raise USvisaException(e, sys)

    def upload_file(self, local_file_path: str, container_name: str, blob_name: str):
        """
        Upload a file to Azure Blob Storage
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
           
            with open(local_file_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        except Exception as e:
            raise USvisaException(e, sys)

    def is_blob_exists(self, container_name: str, blob_name: str) -> bool:
        """
        Check if a blob exists in the container
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            return container_client.get_blob_client(blob_name).exists()
        except Exception as e:
            raise USvisaException(e, sys)