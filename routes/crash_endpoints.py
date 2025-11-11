from fastapi import FastAPI
from utils.logger import log
from utils.traceback_utils import traceback_block


def register_crash_routes(app: FastAPI):

    @app.get("/crash/zerodiv")
    def crash_zero():
        try:
            x = 1 / 0   
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/keyerror")
    def crash_key():
        try:
            d = {"x": 1}
            return d["missing"]
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/typeerror")
    def crash_type():
        try:
            x = "hello"
            return x + 5
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/importerror")
    def crash_import():
        try:
            import non_existing_module
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/attribute")
    def crash_attr():
        try:
            obj = None
            obj.run()
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/file-not-found")
    def crash_missing_file():
        # mhmd: surface missing dependency/config files early
        try:
            with open("non_existent_config.yaml", "r", encoding="utf8") as f:
                return f.read()
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/json-parse")
    def crash_bad_json():
        # mhmd: capture downstream JSON ingestion failures
        import json
        raw_payload = '{"status": "ok", "meta": '  # malformed on purpose
        try:
            return json.loads(raw_payload)
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/sqlite-closed")
    def crash_sqlite_closed():
        # mhmd: expose improper DB lifecycle handling
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.close()
        try:
            conn.execute("SELECT 1")
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/crash/datetime-parse")
    def crash_datetime_parse():
        # mhmd: highlight brittle timestamp parsing paths
        from datetime import datetime
        ts = "2024/31/12 25:61:00"
        try:
            datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except:
            log(traceback_block(), level="ERROR")
            raise
