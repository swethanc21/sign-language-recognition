import sys

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        _, _, traceback=sys.exc_info()
        file_name = traceback.tb_frame.f_code.co_filename
        line_number = traceback.tb_lineno

        error_message=(
            f"Error occurred in file: {file_name}\n"
            f"line number: {line_number}\n"
            f"error message: {error_message}"
        )
        super().__init__(error_message)