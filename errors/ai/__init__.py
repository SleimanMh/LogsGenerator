
"""
AI Error Endpoints - 200 Deep Learning/AI Errors
Covers: PyTorch, TensorFlow, Transformers, Computer Vision, NLP, RNNs, etc.

Organization:
  - preprocessing_errors: Data processing and loading (e_ai_01-20)
  - vision_errors: Computer vision and CNN (e_ai_21-40)
  - embeddings_errors: Embeddings and NLP (e_ai_41-60)
  - autograd_errors: Autograd and RNN (e_ai_61-80)
  - transformers_errors: Transformers and advanced (e_ai_81-100)
  - nextgen_errors: Extended versions across all categories (e_ai_101-200)

API Endpoints:
  - GET /ai/preprocessing (40 errors)
  - GET /ai/vision (40 errors)
  - GET /ai/embeddings (40 errors)
  - GET /ai/autograd (40 errors)
  - GET /ai/transformers (40 errors)
"""

from fastapi import FastAPI
from utils.logger import log
import traceback

# Import all error modules
from . import preprocessing_errors
from . import vision_errors
from . import embeddings_errors
from . import autograd_errors
from . import transformers_errors
from . import nextgen_errors


AI_ERRORS = {}
MODULES = [
    preprocessing_errors,
    vision_errors,
    embeddings_errors,
    autograd_errors,
    transformers_errors,
    nextgen_errors,
]

for module in MODULES:
    for attr_name in dir(module):
        if attr_name.startswith('e_ai_'):
            AI_ERRORS[attr_name] = getattr(module, attr_name)


CATEGORY_RANGES = {
    "preprocessing": [(1, 20), (101, 120)],
    "vision": [(21, 40), (121, 140)],
    "embeddings": [(41, 60), (141, 160)],
    "autograd": [(61, 80), (161, 180)],
    "transformers": [(81, 100), (181, 200)],
}


def _err_id(name: str) -> int:
    try:
        return int(name.split('_')[2])
    except (IndexError, ValueError):
        return -1


def _select_errors(range_pairs):
    selected = {}
    for name, func in AI_ERRORS.items():
        idx = _err_id(name)
        if any(start <= idx <= end for start, end in range_pairs):
            selected[name] = func
    return selected


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


def register_ai_errors(app: FastAPI):
    """Register AI error endpoints"""

    preprocessing_dict = _select_errors(CATEGORY_RANGES["preprocessing"])
    vision_dict = _select_errors(CATEGORY_RANGES["vision"])
    embeddings_dict = _select_errors(CATEGORY_RANGES["embeddings"])
    autograd_dict = _select_errors(CATEGORY_RANGES["autograd"])
    transformers_dict = _select_errors(CATEGORY_RANGES["transformers"])

    @app.get("/ai/preprocessing")
    def ai_preprocessing():
        """Execute AI preprocessing errors (e_ai_01-20 + e_ai_101-120)"""
        return _run_error_category(preprocessing_dict, "preprocessing")

    @app.get("/ai/vision")
    def ai_vision():
        """Execute AI vision errors (e_ai_21-40 + e_ai_121-140)"""
        return _run_error_category(vision_dict, "vision")

    @app.get("/ai/embeddings")
    def ai_embeddings():
        """Execute AI embeddings errors (e_ai_41-60 + e_ai_141-160)"""
        return _run_error_category(embeddings_dict, "embeddings")

    @app.get("/ai/autograd")
    def ai_autograd():
        """Execute AI autograd errors (e_ai_61-80 + e_ai_161-180)"""
        return _run_error_category(autograd_dict, "autograd")

    @app.get("/ai/transformers")
    def ai_transformers():
        """Execute AI transformer errors (e_ai_81-100 + e_ai_181-200)"""
        return _run_error_category(transformers_dict, "transformers")

    @app.get("/ai/run-all")
    def ai_run_all():
        """Execute all AI error scenarios"""
        results = {"total": len(AI_ERRORS), "errors": []}
        failed_count = 0

        for error_name, error_func in sorted(AI_ERRORS.items()):
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

        results["succeeded"] = len(AI_ERRORS) - failed_count
        results["failed"] = failed_count

        return results

    @app.get("/ai/info")
    def ai_info():
        """Get AI errors information"""
        return {
            "class": "ai",
            "total_errors": len(AI_ERRORS),
            "endpoints": {
                "/ai/preprocessing": len(preprocessing_dict),
                "/ai/vision": len(vision_dict),
                "/ai/embeddings": len(embeddings_dict),
                "/ai/autograd": len(autograd_dict),
                "/ai/transformers": len(transformers_dict),
                "/ai/run-all": len(AI_ERRORS)
            }
        }


__all__ = ['register_ai_errors', 'AI_ERRORS']
