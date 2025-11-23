"""
ML Error Endpoints - 200 Machine Learning Errors
Covers: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch Lightning,
imbalanced-learn, and production ML workflows.

Organization:
  - preprocessing_errors: Data handling and preprocessing (ml_err_01-30)
  - training_errors: Model training and fitting (ml_err_31-60)
  - metrics_errors: Metrics and evaluation (ml_err_61-85)
  - advanced_errors: Advanced and complex scenarios (ml_err_86-100)
  - nextgen_errors: Extended scenarios across all categories (ml_err_101-180)
  - propagation_errors: Multi-layer error propagation (ml_err_181-200)

API Endpoints:
  - GET /ml/preprocessing (50 errors)
  - GET /ml/training (50 errors)
  - GET /ml/metrics (45 errors)
  - GET /ml/advanced (35 errors)
  - GET /ml/nextgen (80 errors)
  - GET /ml/propagation (20 errors)
"""

from fastapi import FastAPI
from utils.logger import log
import traceback

from . import preprocessing_errors
from . import training_errors
from . import metrics_errors
from . import advanced_errors
from . import nextgen_errors
from . import propagation_errors


MODULES = [
    preprocessing_errors,
    training_errors,
    metrics_errors,
    advanced_errors,
    nextgen_errors,
    propagation_errors,
]

ML_ERRORS = {}
for module in MODULES:
    for attr in dir(module):
        if attr.startswith("ml_err_"):
            ML_ERRORS[attr] = getattr(module, attr)


CATEGORY_RANGES = {
    "preprocessing": [(1, 30), (101, 120)],
    "training": [(31, 60), (121, 140)],
    "metrics": [(61, 85), (141, 160)],
    "advanced": [(86, 100), (161, 180)],
    "propagation": [(181, 200)],
}


def _err_id(name: str) -> int:
    """Extract numeric ID from error name (e.g., ml_err_001 -> 1)"""
    try:
        parts = name.split('_')
        if len(parts) >= 3:
            return int(parts[2])
        return -1
    except (IndexError, ValueError):
        return -1


def _select_errors(range_pairs):
    selected = {}
    for name, func in ML_ERRORS.items():
        idx = _err_id(name)
        if any(start <= idx <= end for start, end in range_pairs):
            selected[name] = func
    return selected


def _run_error_category(errors_dict: dict, category_name: str) -> dict:
    results = {"category": category_name, "total": len(errors_dict), "errors": []}
    failed = 0

    for error_name, error_func in sorted(errors_dict.items()):
        try:
            error_func()
        except Exception as e:
            error_type = type(e).__name__
            tb_clean = "".join(traceback.format_exc()).replace(
                "Traceback (most recent call last):", ""
            ).strip()

            log(f"[ERROR] {error_name}: {error_type}\n{tb_clean}")
            results["errors"].append({"function": error_name, "type": error_type})
            failed += 1

    results["succeeded"] = len(errors_dict) - failed
    results["failed"] = failed
    return results


def register_ml_errors(app: FastAPI):
    preprocessing_dict = _select_errors(CATEGORY_RANGES["preprocessing"])
    training_dict = _select_errors(CATEGORY_RANGES["training"])
    metrics_dict = _select_errors(CATEGORY_RANGES["metrics"])
    advanced_dict = _select_errors(CATEGORY_RANGES["advanced"])
    nextgen_dict = {k: v for k, v in ML_ERRORS.items() if 101 <= _err_id(k) <= 180}
    propagation_dict = _select_errors(CATEGORY_RANGES["propagation"])

    @app.get("/ml/preprocessing")
    def ml_preprocessing():
        """Execute ML preprocessing errors (ml_err_01-30 + ml_err_101-120)"""
        return _run_error_category(preprocessing_dict, "preprocessing")

    @app.get("/ml/training")
    def ml_training():
        """Execute ML training errors (ml_err_31-60 + ml_err_121-140)"""
        return _run_error_category(training_dict, "training")

    @app.get("/ml/metrics")
    def ml_metrics():
        """Execute ML metrics errors (ml_err_61-85 + ml_err_141-160)"""
        return _run_error_category(metrics_dict, "metrics")

    @app.get("/ml/advanced")
    def ml_advanced():
        """Execute ML advanced errors (ml_err_86-100 + ml_err_161-180)"""
        return _run_error_category(advanced_dict, "advanced")

    @app.get("/ml/nextgen")
    def ml_nextgen():
        """Execute ML nextgen extended errors (ml_err_101-180)"""
        return _run_error_category(nextgen_dict, "nextgen")

    @app.get("/ml/propagation")
    def ml_propagation():
        """Execute ML propagation errors (ml_err_181-200)"""
        return _run_error_category(propagation_dict, "propagation")

    @app.get("/ml/info")
    def ml_info():
        """Get ML errors information"""
        return {
            "class": "ml",
            "total_errors": len(ML_ERRORS),
            "endpoints": {
                "/ml/preprocessing": len(preprocessing_dict),
                "/ml/training": len(training_dict),
                "/ml/metrics": len(metrics_dict),
                "/ml/advanced": len(advanced_dict),
                "/ml/nextgen": len(nextgen_dict),
                "/ml/propagation": len(propagation_dict)
            },
        }


__all__ = ["register_ml_errors", "ML_ERRORS"]
