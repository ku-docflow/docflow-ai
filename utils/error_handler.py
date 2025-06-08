# backend/utils/error_handler.py
from flask import jsonify

def handle_error(error_code: str, message: str, status_code: int):
    """
    Return a dictionary formatted error response with the given parameters.
    """
    response = {
        "error": error_code,
        "message": message
    }
    return response, status_code
