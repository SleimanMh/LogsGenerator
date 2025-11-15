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

    @app.get("/crash/network-timeout")
    def crash_network_timeout():
        import requests
        try:
            requests.get("https://10.255.255.1", timeout=0.001)
        except Exception as e:
            log(str(e), level="ERROR")
            raise


    @app.get("/crash/json-decode")
    def crash_json_decode():
        import json
        try:
            json.loads("{bad json}")
        except Exception as e:
            log(str(e), level="ERROR")
            raise


    @app.get("/crash/db-operational")
    def crash_db_operational():
        import sqlite3
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("SELECT * FROM missing_table")
        except Exception as e:
            log(str(e), level="ERROR")
            raise


    @app.get("/crash/permission-denied")
    def crash_permission():
        try:
            with open("/root/secret.txt", "w") as f:
                f.write("denied")
        except Exception as e:
            log(str(e), level="ERROR")
            raise


    @app.get("/crash/memory-error")
    def crash_memory():
        try:
            x = "x" * (10**9)
        except Exception as e:
            log(str(e), level="ERROR")
            raise


    @app.get("/crash/data-validation")
    def crash_data_validation():
        try:
            data = {"name": None, "age": -5}
            if data["age"] < 0:
                raise ValueError("Age cannot be negative")
        except Exception as e:
            log(str(e), level="ERROR")
            raise


    @app.get("/crash/recursion")
    def crash_recursion():
        try:
            def recurse(): return recurse()
            recurse()
        except Exception as e:
            log(str(e), level="ERROR")
            raise

