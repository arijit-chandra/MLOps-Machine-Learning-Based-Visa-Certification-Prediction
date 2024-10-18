from us_visa_prediction.logger import logging
from us_visa_prediction.exception import VisaException
import sys

try:
    a = 5/0
except Exception as e:
    raise VisaException(e,sys)

# logging.info("Logging setup complete!")