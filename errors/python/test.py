"""
Custom Python Errors (p_err_131 - p_err_135)
Advanced runtime issues: recursion depth, metaclass conflicts,
async misuse, import traps, and bad decorators.
"""

import traceback
from utils.logger import log


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Custom Errors (131 - 135)
# ============================================================

def p_err_131():
    """Race condition: reading a file before it is fully written"""
    import tempfile
    import threading
    import time
    import os

    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    tmp.close()

    def writer():
        time.sleep(0.2)
        with open(path, "w") as f:
            f.write("partial data")

    threading.Thread(target=writer).start()

    with open(path, "r") as f:
        content = f.read().strip()
        if content == "":
            raise RuntimeError("Race condition: file not ready yet")



def p_err_132():
    """Invalid JSON configuration — common real-world parsing failure"""
    import json

    bad_json = """
    {
        "service": "backend",
        "port": 8080,
        "features": ["auth", "logging",  // <-- invalid trailing comma
    }
    """

    config = json.loads(bad_json)


def p_err_133():
    """Simulated database connection failure using invalid connection string"""
    import sqlite3

    conn = sqlite3.connect("/invalid/path/to/database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users")


def p_err_134():
    """Simulated circular-wait deadlock between two threads"""
    import threading
    import time

    lock_a = threading.Lock()
    lock_b = threading.Lock()

    deadlock_detected = False

    def t1():
        nonlocal deadlock_detected
        lock_a.acquire()
        time.sleep(0.1)  # pretend to do work

        if not lock_b.acquire(timeout=0.2):
            deadlock_detected = True

    def t2():
        lock_b.acquire()
        time.sleep(0.1)

        if not lock_a.acquire(timeout=0.2):
            pass  # would deadlock here in real scenario

    # run both threads
    thread1 = threading.Thread(target=t1)
    thread2 = threading.Thread(target=t2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    if deadlock_detected:
        raise RuntimeError("Deadlock detected: circular resource acquisition")

def p_err_135():
    """Missing required environment variable"""
    import os

    required = "API_SECRET_KEY"

    if required not in os.environ:
        raise EnvironmentError(f"Missing required env variable: {required}")

    key = os.environ[required]
    if len(key) < 10:
        raise ValueError("Environment variable API_SECRET_KEY too short")


__all__ = [
    "p_err_131",
    "p_err_132",
    "p_err_133",
    "p_err_134",
    "p_err_135",
]
