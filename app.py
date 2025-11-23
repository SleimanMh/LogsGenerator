from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading
from utils.logger import log

from errors.python import register_python_errors
from errors.ml import register_ml_errors
from errors.ai import register_ai_errors

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown"""
    log("[SYSTEM] Application starting up...")
    log("[SYSTEM] Loading new modular error structure...")
    log("[SYSTEM] Python Errors: 100 (types, strings, I/O, arithmetic, iteration)")
    log("[SYSTEM] ML Errors: 200 (preprocessing, training, metrics, advanced + nextgen + propagation)")
    log("[SYSTEM] AI Errors: 270 (preprocessing, vision, embeddings, autograd, transformers + nextgen + propagation + deeplearning)")
    yield
    log("[SYSTEM] Application shutting down...")


app = FastAPI(lifespan=lifespan)

register_python_errors(app)
register_ml_errors(app)    
register_ai_errors(app)        


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Error Classification Dataset Generator",
        "version": "2.0",
        "description": "Modular error generation system for 3-class ML classification",
        "endpoints": {
            "python_errors": "100 errors (5 categories)",
            "ml_errors": "200 errors (6 categories)",
            "ai_errors": "270 errors (7 categories)",
            "python_info": "/python/info",
            "ml_info": "/ml/info",
            "ai_info": "/ai/info"
        },
        "total_errors": 570,
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
                    "modules": ["preprocessing_errors", "training_errors", "metrics_errors", "advanced_errors", "nextgen_errors", "propagation_errors"],
                    "count": 200
                },
                "ai": {
                    "modules": ["preprocessing_errors", "vision_errors", "embeddings_errors", "autograd_errors", "transformers_errors", "nextgen_errors", "error_propagation", "deeplearning_errors"],
                    "count": 270
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
