"""
Utility functions for error generation
"""

import traceback
from typing import Tuple
from utils.logger import log


def clean_traceback() -> str:
    """Clean and format traceback"""
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


def handle_error(error_name: str, error_func, logger=None) -> Tuple[bool, str]:
    """
    Execute error function and log the result
    
    Args:
        error_name: Name of the error function
        error_func: The error function to execute
        logger: Optional logger function
    
    Returns:
        Tuple of (success, message)
    """
    try:
        error_func()
        return False, "Expected error did not occur"
    except Exception as e:
        error_type = type(e).__name__
        tb = clean_traceback()
        
        if logger:
            logger(f"[ERROR] {error_name}: {error_type}\n{tb}")
        else:
            log(f"[ERROR] {error_name}: {error_type}\n{tb}")
        
        return True, error_type


def register_endpoint(app, path: str, handler_func):
    """Register a FastAPI endpoint"""
    @app.get(path)
    async def endpoint():
        return await handler_func()
    
    return endpoint
