from fastapi import FastAPI
from utils.logger import log
import traceback
from error_propagation.controller import handle_request

def trigger_numpy():
    import numpy as np
    try:
        x = np.array([5])
        np.asscalar(x)
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_numpy_broadcast():
    import numpy as np
    try:
        a = np.ones((3, 3))
        b = np.ones((4,))
        _ = a + b
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_numpy_invalid_index():
    import numpy as np
    try:
        arr = np.array([1, 2, 3])
        _ = arr[10]
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_pandas():
    import pandas as pd
    try:
        df = pd.DataFrame({"a": [1]})
        df = df.append({"a": 2})
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_pandas_merge():
    import pandas as pd
    try:
        df1 = pd.DataFrame({"id": [1], "name": ["A"]})
        df2 = pd.DataFrame({"user": [1], "age": [20]})
        _ = pd.merge(df1, df2, on="id")
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_tensorflow():
    import tensorflow as tf
    try:
        x = tf.ones((5,))
        _ = tf.compat.v1.nn.dropout(x, keep_prob=0.5)
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_tensorflow_shape():
    import tensorflow as tf
    try:
        a = tf.ones((3, 3))
        b = tf.ones((4, 4))
        _ = tf.matmul(a, b)
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_torch():
    import torch
    try:
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(3, dtype=torch.float32)
        _ = a + b
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_torch_cuda():
    import torch
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        _ = torch.randn(5).cuda()
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_sklearn():
    from sklearn.linear_model import LinearRegression
    try:
        model = LinearRegression(normalize=True)
        model.fit([[1], [2], [3]], [1, 2, 3])
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_sklearn_wrong_shape():
    from sklearn.linear_model import LogisticRegression
    try:
        model = LogisticRegression()
        model.fit([[1, 2], [3, 4]], [0])
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_sklearn_invalid_param():
    from sklearn.ensemble import RandomForestClassifier
    try:
        model = RandomForestClassifier(n_estimators="invalid")
        model.fit([[1], [2]], [0, 1])
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_python_zero_div():
    try:
        _ = 1 / 0
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_python_attr_error():
    try:
        x = None
        x.strip()
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_python_key_error():
    try:
        d = {"a": 1}
        _ = d["b"]
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_python_file_not_found():
    try:
        open("this_file_does_not_exist_abc123.txt", "r")
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_python_import_error():
    try:
        import nonexistent_module
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def trigger_python_type_error():
    try:
        _ = "a" + 5
    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise

def trigger_realworld_multifile():
    try:
        bad_request = {
            "user_id": 42,
            "payload": [5, 7] 
        }

        handle_request(bad_request)

    except Exception:
        log(traceback.format_exc(), level="ERROR")
        raise


def register_routes(app: FastAPI):

    @app.get("/trigger/numpy")
    def numpy_api():
        return trigger_numpy()

    @app.get("/trigger/numpy/broadcast")
    def numpy_broadcast_api():
        return trigger_numpy_broadcast()

    @app.get("/trigger/numpy/index")
    def numpy_index_api():
        return trigger_numpy_invalid_index()

    @app.get("/trigger/pandas")
    def pandas_api():
        return trigger_pandas()

    @app.get("/trigger/pandas/merge")
    def pandas_merge_api():
        return trigger_pandas_merge()

    @app.get("/trigger/tensorflow")
    def tensorflow_api():
        return trigger_tensorflow()

    @app.get("/trigger/tensorflow/shape")
    def tensorflow_shape_api():
        return trigger_tensorflow_shape()

    @app.get("/trigger/torch")
    def torch_api():
        return trigger_torch()

    @app.get("/trigger/torch/cuda")
    def torch_cuda_api():
        return trigger_torch_cuda()

    @app.get("/trigger/sklearn")
    def sklearn_api():
        return trigger_sklearn()

    @app.get("/trigger/sklearn/shape")
    def sklearn_shape_api():
        return trigger_sklearn_wrong_shape()

    @app.get("/trigger/sklearn/invalid-param")
    def sklearn_invalid_param_api():
        return trigger_sklearn_invalid_param()

    @app.get("/trigger/python/zero-div")
    def python_zero_api():
        return trigger_python_zero_div()

    @app.get("/trigger/python/attr")
    def python_attr_api():
        return trigger_python_attr_error()

    @app.get("/trigger/python/key")
    def python_key_api():
        return trigger_python_key_error()

    @app.get("/trigger/python/file")
    def python_file_api():
        return trigger_python_file_not_found()

    @app.get("/trigger/python/import")
    def python_import_api():
        return trigger_python_import_error()

    @app.get("/trigger/python/type")
    def python_type_api():
        return trigger_python_type_error()

    @app.get("/trigger/realworld")
    def realworld_api():
        return trigger_realworld_multifile()
