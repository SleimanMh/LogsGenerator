"""
Error Classification System - Organized Error Package
Provides structured error generation for 3-class ML classification

Subpackages:
  - common: Base classes and utilities for error handling
  - python: 100 Python language errors (types, strings, I/O, arithmetic, iteration)
  - ml: 100 Traditional ML errors (preprocessing, training, metrics, advanced)
  - ai: 100 Deep learning AI errors (preprocessing, vision, embeddings, autograd, transformers)
"""

from .common import BaseErrorHandler, ErrorRegistry, clean_traceback, handle_error
from .python import register_python_errors, PYTHON_ERRORS
from .ml import register_ml_errors, ML_ERRORS
from .ai import register_ai_errors, AI_ERRORS

__all__ = [
    'BaseErrorHandler',
    'ErrorRegistry',
    'clean_traceback',
    'handle_error',
    'register_python_errors',
    'register_ml_errors',
    'register_ai_errors',
    'PYTHON_ERRORS',
    'ML_ERRORS',
    'AI_ERRORS',
]
