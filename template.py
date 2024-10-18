import os
from pathlib import Path

project_name = "us_visa_prediction"

list_of_files = [

    f"{project_name}/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",  
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/pipline/__init__.py",
    f"{project_name}/pipline/training_pipeline.py",
    f"{project_name}/pipline/prediction_pipeline.py",
    "notebook/EDA.ipynb",
    "notebook/Feature_Engineering_Model_Training.ipynb",
    "notebook/mongodb_demo.ipynb",
    "templates/demo.html",
    "static/css/style.css",
    "requirements.txt",
    "app.py",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
    "Dockerfile",
    ".dockerignore",

]

def create_structure(list_of_files):
    for filepath in list_of_files:
        try:
            # Ensure filepath is a Path object
            filepath = Path(filepath)

            # Extract directory and file name
            filedir = filepath.parent

            # Create directory if it doesn't exist
            if filedir and not filedir.exists():
                filedir.mkdir(parents=True, exist_ok=True)
                print(f"Creating directory: {filedir}")

            # Check if file exists or is empty, and create if necessary
            if not filepath.exists() or filepath.stat().st_size == 0:
                filepath.touch(exist_ok=True)
                print(f"Creating an empty file at: {filepath}")
            else:
                print(f"File already exists and is not empty: {filepath}")

        except OSError as e:
            print(f"Error processing file {filepath}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


create_structure(list_of_files)

