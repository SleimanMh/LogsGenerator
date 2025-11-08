import time
from utils.logger import log
from utils.traceback_utils import traceback_block

def start_trainer():
    while True:
        log("[TRAINER] Running background ML task...")
        time.sleep(4)
        try:
            raise RuntimeError("Background trainer simulated failure")
        except:
            log(traceback_block(), level="ERROR")
            time.sleep(4)