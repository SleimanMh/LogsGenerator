from fastapi import FastAPI
import threading
from utils.logger import log
from background.trainer import start_trainer
from routes.crash_endpoints import register_crash_routes
from routes.ml_endpoints import register_ml_routes
from routes.deprecation_triggers import register_deprecation_routes

app = FastAPI()


@app.on_event("startup")
def startup_event():
    log("[SYSTEM] Application starting up...")
    t = threading.Thread(target=start_trainer, daemon=True)
    t.start()


# register_crash_routes(app)
# register_ml_routes(app)
register_deprecation_routes(app)
