from fastapi import FastAPI
from utils.logger import log
from utils.traceback_utils import traceback_block


def register_crash_routes(app: FastAPI):

    # ---------------------------------------------------------
    # Base 15 exceptions you already had (renamed for uniformity)
    # ---------------------------------------------------------

    def e_zero_division():
        try: 1 / 0
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_key_error():
        try:
            d = {"x": 1}
            _ = d["missing"]
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_type_error():
        try: "hello" + 5
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_import_error():
        try: import non_existing_module
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_attribute_error():
        try:
            obj = None
            obj.run()
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_file_not_found():
        try: open("non_existent_config.yaml", "r")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_json_parse():
        import json
        try: json.loads('{"status": "ok", "meta": ')
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_sqlite_closed():
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.close()
        try: conn.execute("SELECT 1")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_datetime_parse():
        from datetime import datetime
        try: datetime.strptime("2024/31/12 25:61:00", "%Y-%m-%d %H:%M:%S")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_network_timeout():
        import requests
        try: requests.get("https://10.255.255.1", timeout=0.0001)
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_json_decode():
        import json
        try: json.loads("{bad json}")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_db_operational():
        import sqlite3
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("SELECT * FROM missing_table")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_permission_denied():
        try:
            with open("/root/secret.txt", "w") as f: f.write("x")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_memory_error():
        try: "x" * (10**12)
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_value_error_custom():
        try:
            data = {"name": None, "age": -5}
            if data["age"] < 0:
                raise ValueError("Age cannot be negative")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_recursion():
        try:
            def r(): return r()
            r()
        except:
            log(traceback_block(), level="ERROR"); raise

    # ---------------------------------------------------------
    # ADDITIONAL 20 NEW CRASH / SYSTEM / PYTHON ERRORS
    # ---------------------------------------------------------

    def e_env_missing():
        import os
        try: os.environ["MISSING_ENV"]
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_index_error():
        try:
            arr = [1, 2, 3]
            _ = arr[10]
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_os_error():
        import os
        try: os.remove("file_does_not_exist_123.txt")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_timeout_custom():
        import time
        try:
            raise TimeoutError("Artificial timeout reached during operation")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_unicode_decode():
        try: b"\xff\xff".decode("utf-8")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_regex_error():
        import re
        try: re.compile("[Unclosed(")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_subprocess_fail():
        import subprocess
        try: subprocess.check_output(["nonexistent_cmd_123"])
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_socket_error():
        import socket
        try:
            s = socket.socket()
            s.connect(("256.256.256.256", 80))
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_json_type_error():
        import json
        try: json.dumps(set([1,2,3]))  # set is not serializable
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_utf8_encode():
        try: "hello 😊".encode("ascii")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_divmod_error():
        try: divmod(5, 0)
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_format_error():
        try: "{} {} {}".format(1, 2)  # missing arg
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_cast_error():
        try: int("hello")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_float_cast():
        try: float("not_a_number")
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_path_error():
        import pathlib
        try: pathlib.Path("/invalid\0path").exists()
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_bytes_index():
        try:
            x = b"abc"
            _ = x[10]
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_module_attribute():
        import math
        try: math.pi = 50
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_json_key_error():
        import json
        try:
            obj = json.loads('{"a": 1}')
            _ = obj["b"]
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_wrong_format():
        try: "{x}".format()  # missing x
        except:
            log(traceback_block(), level="ERROR"); raise

    def e_enum_error():
        from enum import Enum
        try:
            class X(Enum): A = 1
            X(999)  # invalid value
        except:
            log(traceback_block(), level="ERROR"); raise


    # ---------------------------------------------------------
    #  ONE ROUTE THAT TRIGGERS ALL CRASH EXCEPTIONS
    # ---------------------------------------------------------

    @app.get("/crash/run-all")
    def crash_run_all():

        exceptions = [
            e_zero_division, e_key_error, e_type_error, e_import_error,
            e_attribute_error, e_file_not_found, e_json_parse, e_sqlite_closed,
            e_datetime_parse, e_network_timeout, e_json_decode, e_db_operational,
            e_permission_denied, e_memory_error, e_value_error_custom, e_recursion,

            # NEW 20:
            e_env_missing, e_index_error, e_os_error, e_timeout_custom,
            e_unicode_decode, e_regex_error, e_subprocess_fail, e_socket_error,
            e_json_type_error, e_utf8_encode, e_divmod_error, e_format_error,
            e_cast_error, e_float_cast, e_path_error, e_bytes_index,
            e_module_attribute, e_json_key_error, e_wrong_format, e_enum_error,
        ]

        for fn in exceptions:
            try: fn()
            except Exception:
                pass  # continue to next exception

        return {"status": "done", "executed": len(exceptions)}
