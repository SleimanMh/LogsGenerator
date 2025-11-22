"""
ML Error Endpoints - 180+ Machine Learning Errors
Covers: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch Lightning,
imbalanced-learn, and production ML workflows.

Organization:
  - preprocessing_errors: Data handling and preprocessing (ml_err_01-30)
  - training_errors: Model training and fitting (ml_err_31-60)
  - metrics_errors: Metrics and evaluation (ml_err_61-85)
  - advanced_errors: Advanced and complex scenarios (ml_err_86-100)
  - nextgen_errors: Extended scenarios across all categories (ml_err_101-180)

API Endpoints:
  - GET /ml/preprocessing (50 errors)
  - GET /ml/training (50 errors)
  - GET /ml/metrics (45 errors)
  - GET /ml/advanced (35 errors)
"""

from fastapi import FastAPI
from utils.logger import log
import traceback

from . import preprocessing_errors
from . import training_errors
from . import metrics_errors
from . import advanced_errors
from . import nextgen_errors


MODULES = [
    preprocessing_errors,
    training_errors,
    metrics_errors,
    advanced_errors,
    nextgen_errors,
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
}


def _err_id(name: str) -> int:
    try:
        return int(name.replace("ml_err_", ""))
    except ValueError:
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
                "/ml/preprocessing": len(preprocessing_dict),
                "/ml/training": len(training_dict),
                "/ml/metrics": len(metrics_dict),
                "/ml/advanced": len(advanced_dict),
                "/ml/run-all": len(ML_ERRORS),
            },
        }


__all__ = ["register_ml_errors", "ML_ERRORS"]
