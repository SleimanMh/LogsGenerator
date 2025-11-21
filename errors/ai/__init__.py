"""
AI Error Endpoints - 100+ Deep Learning/AI Errors
Covers: PyTorch, TensorFlow, Transformers, Computer Vision, NLP, RNNs, etc.

Organization:
  - preprocessing_errors: Data processing and loading (e_ai_01-20)
  - vision_errors: Computer vision and CNN (e_ai_21-40)
  - embeddings_errors: Embeddings and NLP (e_ai_41-60)
  - autograd_errors: Autograd and RNN (e_ai_61-80)
  - transformers_errors: Transformers and advanced (e_ai_81-100)

API Endpoints:
  - GET /ai/preprocessing (20 errors)
  - GET /ai/vision (20 errors)
  - GET /ai/embeddings (20 errors)
  - GET /ai/autograd (20 errors)
  - GET /ai/transformers (20 errors)
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


# Collect all error functions
AI_ERRORS = {}

# Populate registry
for module in [preprocessing_errors, vision_errors, embeddings_errors, autograd_errors, transformers_errors]:
    for attr_name in dir(module):
        if attr_name.startswith('e_ai_'):
            AI_ERRORS[attr_name] = getattr(module, attr_name)


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
    
    # Extract errors by category
    preprocessing_dict = {k: v for k, v in AI_ERRORS.items() if int(k.split('_')[2]) <= 20}
    vision_dict = {k: v for k, v in AI_ERRORS.items() if 21 <= int(k.split('_')[2]) <= 40}
    embeddings_dict = {k: v for k, v in AI_ERRORS.items() if 41 <= int(k.split('_')[2]) <= 60}
    autograd_dict = {k: v for k, v in AI_ERRORS.items() if 61 <= int(k.split('_')[2]) <= 80}
    transformers_dict = {k: v for k, v in AI_ERRORS.items() if int(k.split('_')[2]) >= 81}
    
    @app.get("/ai/preprocessing")
    def ai_preprocessing():
        """Execute 20 AI preprocessing errors (e_ai_01-20)"""
        return _run_error_category(preprocessing_dict, "preprocessing")
    
    @app.get("/ai/vision")
    def ai_vision():
        """Execute 20 AI vision errors (e_ai_21-40)"""
        return _run_error_category(vision_dict, "vision")
    
    @app.get("/ai/embeddings")
    def ai_embeddings():
        """Execute 20 AI embeddings errors (e_ai_41-60)"""
        return _run_error_category(embeddings_dict, "embeddings")
    
    @app.get("/ai/autograd")
    def ai_autograd():
        """Execute 20 AI autograd errors (e_ai_61-80)"""
        return _run_error_category(autograd_dict, "autograd")
    
    @app.get("/ai/transformers")
    def ai_transformers():
        """Execute 20 AI transformers errors (e_ai_81-100)"""
        return _run_error_category(transformers_dict, "transformers")
    
    @app.get("/ai/run-all")
    def ai_run_all():
        """Execute all 100 AI error scenarios"""
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
                "/ai/preprocessing": 20,
                "/ai/vision": 20,
                "/ai/embeddings": 20,
                "/ai/autograd": 20,
                "/ai/transformers": 20,
                "/ai/run-all": 100
            }
        }


__all__ = ['register_ai_errors', 'AI_ERRORS']
