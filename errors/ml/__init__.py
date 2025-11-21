"""
ML Error Endpoints - 100+ Machine Learning Errors
Covers: scikit-learn, XGBoost, LightGBM, ensemble methods, preprocessing,
validation, metrics, and traditional ML operations

Organization:
  - preprocessing_errors: Data handling and preprocessing (ml_err_01-30)
  - training_errors: Model training and fitting (ml_err_31-60)
  - metrics_errors: Metrics and evaluation (ml_err_61-85)
  - advanced_errors: Advanced and complex scenarios (ml_err_86-100)

API Endpoints:
  - GET /ml/preprocessing (30 errors)
  - GET /ml/training (30 errors)
  - GET /ml/metrics (25 errors)
  - GET /ml/advanced (15 errors)
"""

from fastapi import FastAPI
from utils.logger import log
import traceback

# Import ML error modules
from . import preprocessing_errors
from . import training_errors
from . import metrics_errors
from . import advanced_errors


# -------------------------------------------------------------------
# Helper: Extract integer error index from function name safely
# ml_err_XX → XX (int)
# -------------------------------------------------------------------
def _err_id(name: str) -> int:
    """Extract error index from 'ml_err_XX' safely."""
    try:
        return int(name.replace("ml_err_", ""))
    except ValueError:
        return -1  # invalid names won't be included


# Collect all ML error functions dynamically
ML_ERRORS = {}

for module in [preprocessing_errors, training_errors, metrics_errors, advanced_errors]:
    for attr in dir(module):
        if attr.startswith("ml_err_"):
            ML_ERRORS[attr] = getattr(module, attr)


# -------------------------------------------------------------------
# Execute an error category and return structured results
# -------------------------------------------------------------------
def _run_error_category(errors_dict: dict, category_name: str) -> dict:
    results = {"category": category_name, "total": len(errors_dict), "errors": []}
    failed_count = 0

    for error_name, error_func in sorted(errors_dict.items()):
        try:
            error_func()
        except Exception as e:
            error_type = type(e).__name__
            tb_clean = "".join(traceback.format_exc()).replace(
                "Traceback (most recent call last):", ""
            ).strip()

            log(f"[ERROR] {error_name}: {error_type}\n{tb_clean}")

            results["errors"].append({
                "function": error_name,
                "type": error_type
            })
            failed_count += 1

    results["succeeded"] = len(errors_dict) - failed_count
    results["failed"] = failed_count
    return results


# -------------------------------------------------------------------
# Register endpoints
# -------------------------------------------------------------------
def register_ml_errors(app: FastAPI):

    # Correct category division
    preprocessing_dict = {k: v for k, v in ML_ERRORS.items() if 1 <= _err_id(k) <= 30}
    training_dict      = {k: v for k, v in ML_ERRORS.items() if 31 <= _err_id(k) <= 60}
    metrics_dict       = {k: v for k, v in ML_ERRORS.items() if 61 <= _err_id(k) <= 85}
    advanced_dict      = {k: v for k, v in ML_ERRORS.items() if 86 <= _err_id(k) <= 100}

    @app.get("/ml/preprocessing")
    def ml_preprocessing():
        return _run_error_category(preprocessing_dict, "preprocessing")

    @app.get("/ml/training")
    def ml_training():
        return _run_error_category(training_dict, "training")

    @app.get("/ml/metrics")
    def ml_metrics():
        return _run_error_category(metrics_dict, "metrics")

    @app.get("/ml/advanced")
    def ml_advanced():
        return _run_error_category(advanced_dict, "advanced")

    @app.get("/ml/run-all")
    def ml_run_all():
        results = {"total": len(ML_ERRORS), "errors": []}
        failed = 0

        for name, func in sorted(ML_ERRORS.items()):
            try:
                func()
            except Exception as e:
                error_type = type(e).__name__
                tb_clean = "".join(traceback.format_exc()).replace(
                    "Traceback (most recent call last):", ""
                ).strip()

                log(f"[ERROR] {name}: {error_type}\n{tb_clean}")
                results["errors"].append({"function": name, "type": error_type})
                failed += 1

        results["succeeded"] = len(ML_ERRORS) - failed
        results["failed"] = failed
        return results

    @app.get("/ml/info")
    def ml_info():
        return {
            "class": "ml",
            "total_errors": len(ML_ERRORS),
            "endpoints": {
                "/ml/preprocessing": 30,
                "/ml/training": 30,
                "/ml/metrics": 25,
                "/ml/advanced": 15,
                "/ml/run-all": 100
            }
        }


__all__ = ["register_ml_errors", "ML_ERRORS"]
