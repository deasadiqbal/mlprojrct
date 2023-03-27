import sys 
import logging 

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail
    error_msg = f"Error: {error} at line {exc_tb.tb_lineno} in {exc_tb.tb_frame.f_code.co_filename}"
    return error_msg

class ExceptionHandler(Exception):
    def __init__(self, error, error_detail: sys):
        self.error = error
        self.error_detail = error_detail
        logging.error(error_message_detail(self.error, self.error_detail))

