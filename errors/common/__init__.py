"""
Common utilities for error generation and handling
"""

from .base import BaseErrorHandler, ErrorRegistry
from .utils import clean_traceback, handle_error

__all__ = [
    'BaseErrorHandler',
    'ErrorRegistry',
    'clean_traceback',
    'handle_error',
]
