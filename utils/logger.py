import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "service.log")


def log(msg, level="INFO"):
    with open(LOG_FILE, "a", encoding="utf8") as f:
        f.write(f"{datetime.now().isoformat()} [{level}] {msg}\n")