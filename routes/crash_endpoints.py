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