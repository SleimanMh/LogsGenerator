"""
Python Error Endpoints - 100+ Basic Python Errors
Covers: Type errors, I/O, string operations, collections, file handling,
system operations, and basic programming logic errors

Organization:
  - types_errors: Type and data structure errors (p_err_01-25)
  - string_errors: String and encoding errors (p_err_26-45)
  - io_errors: I/O and file errors (p_err_46-65)
  - arithmetic_errors: Arithmetic and math errors (p_err_66-85)
  - iteration_errors: Iteration and control flow errors (p_err_86-100)

API Endpoints:
  - GET /python/types (25 errors)
  - GET /python/strings (20 errors)
  - GET /python/io (20 errors)
  - GET /python/arithmetic (20 errors)
  - GET /python/iteration (15 errors)
"""

from fastapi import FastAPI
from utils.logger import log
import traceback

# Import all error modules
from . import types_errors
from . import string_errors
from . import io_errors
from . import arithmetic_errors
from . import iteration_errors


# Collect all error functions
PYTHON_ERRORS = {}

# Populate registry
for module in [types_errors, string_errors, io_errors, arithmetic_errors, iteration_errors]:
    for attr_name in dir(module):
        if attr_name.startswith('p_err_'):
            PYTHON_ERRORS[attr_name] = getattr(module, attr_name)


def _run_error_category(errors_dict: dict, category_name: str) -> dict:
    """Execute errors from a category and return results"""
    results = {"category": category_name, "total": len(errors_dict), "errors": []}
    failed_count = 0
    
    for error_name, error_func in sorted(errors_dict.items()):
        try:
            error_func()
        except Exception as e:
            error_type = type(e).__name__
            tb = "".join(traceback.format_exc())
            tb_clean = tb.replace("Traceback (most recent call last):", "").strip()
            
            log(f"[ERROR] {error_name}: {error_type}\n{tb_clean}")
            results["errors"].append({
                "function": error_name,
                "type": error_type
            })
            failed_count += 1
    
    results["succeeded"] = len(errors_dict) - failed_count
    results["failed"] = failed_count
    
    return results


def register_python_errors(app: FastAPI):
    """Register Python error endpoints"""
    
    # Extract errors by module
    types_dict = {
        k: v for k, v in PYTHON_ERRORS.items()
        if k.startswith("p_err_") and 1 <= int(k.split('_')[2]) <= 25
    }
    strings_dict = {
        k: v for k, v in PYTHON_ERRORS.items()
        if k.startswith("p_err_") and 26 <= int(k.split('_')[2]) <= 45
    }
    io_dict = {
        k: v for k, v in PYTHON_ERRORS.items()
        if k.startswith("p_err_") and 46 <= int(k.split('_')[2]) <= 65
    }
    arithmetic_dict = {
        k: v for k, v in PYTHON_ERRORS.items()
        if k.startswith("p_err_") and 66 <= int(k.split('_')[2]) <= 85
    }
    iteration_dict = {
        k: v for k, v in PYTHON_ERRORS.items()
        if k.startswith("p_err_") and 86 <= int(k.split('_')[2]) <= 100
    }
    
    @app.get("/python/types")
    def python_types():
        """Execute 25 Python type and data structure errors (p_err_01-25)"""
        return _run_error_category(types_dict, "types")
    
    @app.get("/python/strings")
    def python_strings():
        """Execute 20 Python string and encoding errors (p_err_26-45)"""
        return _run_error_category(strings_dict, "strings")
    
    @app.get("/python/io")
    def python_io():
        """Execute 20 Python I/O and file errors (p_err_46-65)"""
        return _run_error_category(io_dict, "io")
    
    @app.get("/python/arithmetic")
    def python_arithmetic():
        """Execute 20 Python arithmetic and math errors (p_err_66-85)"""
        return _run_error_category(arithmetic_dict, "arithmetic")
    
    @app.get("/python/iteration")
    def python_iteration():
        """Execute 15 Python iteration and control flow errors (p_err_86-100)"""
        return _run_error_category(iteration_dict, "iteration")
    
    @app.get("/python/info")
    def python_info():
        """Get Python errors information"""
        return {
            "class": "python",
            "total_errors": len(PYTHON_ERRORS),
            "endpoints": {
                "/python/types": 25,
                "/python/strings": 20,
                "/python/io": 20,
                "/python/arithmetic": 20,
                "/python/iteration": 15
            }
        }


__all__ = ['register_python_errors', 'PYTHON_ERRORS']
