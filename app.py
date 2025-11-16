from fastapi import FastAPI
import threading
from utils.logger import log
from routes.crash_endpoints import register_crash_routes
from routes.ml_endpoints import register_ml_routes
from routes.triggers import register_routes
from routes.ai_endpoints import register_ai_routes

app = FastAPI()


@app.on_event("startup")
def startup_event():
    log("[SYSTEM] Application starting up...")


register_crash_routes(app)
register_ml_routes(app)
register_routes(app)
register_ai_routes(app)
