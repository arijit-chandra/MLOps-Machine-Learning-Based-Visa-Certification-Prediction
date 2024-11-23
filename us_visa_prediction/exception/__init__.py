import sys
import logging

class USvisaException(Exception):
    """
    Custom exception class for US visa prediction application that provides detailed error information.
    """
   
    def __init__(self, error_message: str, error_detail: sys = None):
        """
        Initialize the custom exception with detailed error information.
       
        Args:
            error_message: Human-readable error message
            error_detail: System information for error traceback (default: None)
        """
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)
        logging.error(self.error_message)  # Log the detailed error message
   
    def _get_detailed_error_message(self, error_message: str, error_detail: sys = None) -> str:
        """
        Generate a detailed error message including file name and line number.
       
        Args:
            error_message: Base error message
            error_detail: System information for error traceback
           
        Returns:
            Formatted error message with file and line information
        """
        # Get exception traceback
        exc_tb = error_detail.exc_info()[2] if error_detail else sys.exc_info()[2]
       
        if exc_tb:
            # Extract file name and line number from traceback
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown File"
            line_number = "Unknown Line"
       
        # Format the detailed error message
        error_message = (
            f"Error occurred in file: [{file_name}] at line number: [{line_number}] "
            f"error message: [{str(error_message)}]"
        )
       
        return error_message
   
    def __str__(self) -> str:
        """
        String representation of the exception.
       
        Returns:
            The detailed error message
        """
        return self.error_message
   
    def __repr__(self) -> str:
        """
        Official string representation of the exception.
       
        Returns:
            Class name with error message
        """
        return f"USvisaException: {self.error_message}"

# Make the exception class available at the module level
__all__ = ['USvisaException']