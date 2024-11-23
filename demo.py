
from us_visa_prediction.pipline.training_pipeline import TrainPipeline

def main():
    try:
        # Create training pipeline object
        pipeline = TrainPipeline()
        
        # Run the training pipeline
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")

if __name__ == "__main__":
    main()
# from us_visa_prediction.logger import logging
# from us_visa_prediction.exception import VisaException
# import sys

# try:
#     a = 5/0
# except Exception as e:
#     raise VisaException(e,sys)

# logging.info("Logging setup complete!")

# import os
# from us_visa_prediction.constants import *

# url = os.getenv(MONGODB_URL_KEY)
# print("url is: ",url)
