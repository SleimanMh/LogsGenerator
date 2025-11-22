from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading
from utils.logger import log

# Import new modular error classes
from errors.python import register_python_errors
from errors.ml import register_ml_errors
from errors.ai import register_ai_errors

# Import legacy error classes (for backward compatibility)
from routes.crash_endpoints import register_crash_routes
from routes.ml_endpoints import register_ml_routes
from routes.triggers import register_routes
from routes.ai_endpoints import register_ai_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown"""
    # Startup
    log("[SYSTEM] Application starting up...")
    log("[SYSTEM] Loading new modular error structure...")
    log("[SYSTEM] Python Errors: 100 (types, strings, I/O, arithmetic, iteration)")
    log("[SYSTEM] ML Errors: 100 (preprocessing, training, metrics, advanced)")
    log("[SYSTEM] AI Errors: 200 (preprocessing, vision, embeddings, autograd, transformers + extended)")
    yield
    # Shutdown
    log("[SYSTEM] Application shutting down...")


app = FastAPI(lifespan=lifespan)


# ============================================================
# Register New Modular Error Classes
# ============================================================
register_python_errors(app)     # 100 Python language errors
register_ml_errors(app)         # 100 Traditional ML errors (sklearn, xgboost, etc)
register_ai_errors(app)         # 200 Deep learning AI errors (torch, tf, transformers + extended)

# ============================================================
# Register Legacy Error Classes (Backward Compatibility)
# ============================================================
# Uncomment to enable legacy endpoints:
# register_crash_routes(app)    # Legacy general errors
# register_ml_routes(app)       # Legacy ML framework errors
# register_routes(app)          # Legacy trigger scenarios
# register_ai_endpoints(app)    # Legacy AI errors


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Error Classification Dataset Generator",
        "version": "2.0",
        "description": "Modular error generation system for 3-class ML classification",
        "endpoints": {
            "python_errors": "/python/run-all (100 errors)",
            "ml_errors": "/ml/run-all (100 errors)",
            "ai_errors": "/ai/run-all (200 errors)",
            "python_info": "/python/info",
            "ml_info": "/ml/info",
            "ai_info": "/ai/info",
            "category_errors": "/python/by-category/{types|strings|io|arithmetic|iteration}",
        },
        "total_errors": 400,
        "classes": ["python", "ml", "ai"],
        "documentation": "See /docs for Swagger UI"
    }


@app.get("/all")
def run_all_errors():
    """Coordinate running all error scenarios"""
    return {
        "message": "Error generation endpoints",
        "description": "Run each endpoint to generate errors from that class",
        "endpoints": {
            "python": "/python/run-all",
            "ml": "/ml/run-all",
            "ai": "/ai/run-all"
        },
        "workflow": [
            "1. Run all endpoints to generate error logs",
            "2. Execute: python error_log_processor.py",
            "3. Execute: python train_classifier.py"
        ]
    }


@app.get("/structure")
def api_structure():
    """Display modular structure"""
    return {
        "package_structure": {
            "errors": {
                "common": ["base.py", "utils.py"],
                "python": {
                    "modules": ["types_errors", "string_errors", "io_errors", "arithmetic_errors", "iteration_errors"],
                    "count": 100
                },
                "ml": {
                    "modules": ["preprocessing_errors", "training_errors", "metrics_errors", "advanced_errors"],
                    "count": 100
                },
                "ai": {
                    "modules": ["preprocessing_errors", "vision_errors", "embeddings_errors", "autograd_errors", "transformers_errors", "nextgen_errors"],
                    "count": 200
                }
            }
        },
        "design_patterns": [
            "Each error class organized into thematic submodules",
            "Implicit exception raising (no explicit raise statements)",
            "Multi-layer error propagation through functions",
            "Clean imports via __init__.py files",
            "No circular dependencies"
        ]
    }
