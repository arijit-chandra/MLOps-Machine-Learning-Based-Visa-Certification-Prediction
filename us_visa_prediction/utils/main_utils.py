import os
import sys
import dill
import yaml
import numpy as np
from pandas import DataFrame
from us_visa_prediction.exception import VisaException
from us_visa_prediction.logger import logging


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its content as a dictionary.
    :param file_path: str path to the YAML file
    :return: dict contents of the YAML file
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VisaException(f"Error reading YAML file: {file_path}", sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file. Optionally, replace the file if it already exists.
    :param file_path: str path to the YAML file
    :param content: object content to write into the file
    :param replace: bool whether to replace the existing file
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise VisaException(f"Error writing to YAML file: {file_path}", sys) from e


def load_object(file_path: str) -> object:
    """
    Load a Python object from a file using dill.
    :param file_path: str path to the file containing the object
    :return: object loaded from the file
    """
    logging.info(f"Loading object from {file_path}")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        raise VisaException(f"Error loading object from file: {file_path}", sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save a numpy array to a file.
    :param file_path: str path to save the numpy array
    :param array: np.array to be saved
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info(f"Numpy array saved successfully at {file_path}")
    except Exception as e:
        raise VisaException(f"Error saving numpy array to file: {file_path}", sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load a numpy array from a file.
    :param file_path: str path to the numpy array file
    :return: np.array loaded from the file
    """
    try:
        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
        logging.info(f"Numpy array loaded successfully from {file_path}")
        return array
    except Exception as e:
        raise VisaException(f"Error loading numpy array from file: {file_path}", sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using dill.
    :param file_path: str path to save the object
    :param obj: object to be saved
    """
    logging.info(f"Saving object to {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise VisaException(f"Error saving object to file: {file_path}", sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop columns from a pandas DataFrame.
    :param df: pandas DataFrame from which to drop columns
    :param cols: list of column names to drop
    :return: DataFrame with specified columns removed
    """
    logging.info(f"Dropping columns: {cols}")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info(f"Columns {cols} dropped successfully")
        return df
    except Exception as e:
        raise VisaException(f"Error dropping columns {cols} from DataFrame", sys) from e
