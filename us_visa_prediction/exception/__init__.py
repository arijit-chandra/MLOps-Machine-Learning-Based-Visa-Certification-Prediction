import sys
import logging

class VisaException(Exception):
    def __init__(self, error_message, error_detail:sys=None):
        """
        Custom exception to capture detailed error information.
        :param error_message: The message to display or log.
        :param error_detail: Optional sys argument to capture error details (traceback).
        """
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)

        # # Logging the error message
        logging.error(self.error_message)

    def error_message_detail(self, error, error_detail:sys=None):
        """
        Captures the error message with traceback details, including file name and line number.
        """
        exc_tb = error_detail.exc_info()[2] if error_detail else sys.exc_info()[2]
        
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "N/A"
            line_number = "N/A"
        
        error_message = f"Error in script: [{file_name}], line number: [{line_number}], message: [{str(error)}]"
        return error_message

    def __str__(self):
        return self.error_message
