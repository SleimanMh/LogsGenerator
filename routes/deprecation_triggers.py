from fastapi import FastAPI
from utils.logger import log
from utils.traceback_utils import traceback_block

# -----------------------------
# Deprecation Trigger Functions
# -----------------------------

def trigger_numpy_deprecation():
    import numpy as np
    try:
        x = np.array([5])
        np.asscalar(x)  # Deprecated / Removed
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def trigger_pandas_deprecation():
    import pandas as pd
    try:
        df = pd.DataFrame({"a": [1]})
        df = df.append({"a": 2})  # Deprecated / Removed API
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def trigger_tf_deprecation():
    import tensorflow as tf
    try:
        x = tf.ones((5,))
        # Deprecated TF1 API inside TF2
        y = tf.compat.v1.nn.dropout(x, keep_prob=0.5)
        return str(y)
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def trigger_torch_deprecation():
    import torch
    try:
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(3, dtype=torch.float32)
        c = a + b  # dtype warning / type mismatch errors
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def trigger_sklearn_deprecation():
    from sklearn.linear_model import LinearRegression
    try:
        # normalize=True is deprecated in sklearn
        model = LinearRegression(normalize=True)
        model.fit([[1], [2], [3]], [1, 2, 3])
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def trigger_python_deprecation():
    import warnings
    import imp
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            imp.find_module("os")  # Deprecated since Python 3.4
            if w:
                raise DeprecationWarning(str(w[-1].message))
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def trigger_removed_mapping():
    try:
        import collections
        collections.Mapping  # Removed in Python 3.10
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


# mhmd: ensure removed NumPy scalar aliases raise visibility
def trigger_numpy_alias_deprecation():
    import numpy as np
    try:
        _ = np.bool  # removed alias, raises AttributeError on modern NumPy
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


# mhmd: highlight legacy pandas containers that were removed
def trigger_pandas_panel_deprecation():
    import pandas as pd
    try:
        pd.Panel({"a": [1, 2, 3]})
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


# mhmd: flag TensorFlow code still referencing tf.contrib
def trigger_tf_contrib_deprecation():
    import tensorflow as tf
    try:
        _ = tf.contrib.slim
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


# mhmd: detect sklearn builds importing deprecated externals.joblib
def trigger_sklearn_joblib_deprecation():
    try:
        from sklearn.externals import joblib  # noqa: F401
    except Exception:
        import traceback
        return "Traceback (most recent call last):\n" + "".join(traceback.format_exc())


def register_deprecation_routes(app: FastAPI):

    @app.get("/deprecation/numpy")
    def dep_numpy():
        out = trigger_numpy_deprecation()
        log(out, level="WARNING")
        return {"status": "numpy_deprecation_triggered"}
  
    @app.get("/deprecation/pandas")
    def dep_pandas():
        out = trigger_pandas_deprecation()
        log(out, level="WARNING")
        return {"status": "pandas_deprecation_triggered"}

    @app.get("/deprecation/tensorflow")
    def dep_tf():
        out = trigger_tf_deprecation()
        log(out, level="WARNING")
        return {"status": "tensorflow_deprecation_triggered"}

    @app.get("/deprecation/torch")
    def dep_torch():
        out = trigger_torch_deprecation()
        log(out, level="WARNING")
        return {"status": "torch_deprecation_triggered"}

    @app.get("/deprecation/sklearn")
    def dep_sklearn():
        out = trigger_sklearn_deprecation()
        log(out, level="WARNING")
        return {"status": "sklearn_deprecation_triggered"}

    @app.get("/deprecation/python")
    def dep_python():
        out = trigger_python_deprecation()
        log(out, level="WARNING")
        return {"status": "python_deprecation_triggered"}

    @app.get("/deprecation/collections-mapping")
    def dep_removed_mapping():
        out = trigger_removed_mapping()
        log(out, level="WARNING")
        return {"status": "collections_mapping_removed_triggered"}

    @app.get("/deprecation/numpy-bool-alias")
    def dep_numpy_alias():
        # mhmd: watch for builds still referencing np.bool / np.int
        out = trigger_numpy_alias_deprecation()
        log(out, level="WARNING")
        return {"status": "numpy_alias_deprecation_triggered"}

    @app.get("/deprecation/pandas-panel")
    def dep_pandas_panel():
        # mhmd: catch teams still shipping Panel-based logic
        out = trigger_pandas_panel_deprecation()
        log(out, level="WARNING")
        return {"status": "pandas_panel_deprecation_triggered"}

    @app.get("/deprecation/tf-contrib")
    def dep_tf_contrib():
        # mhmd: surface TF1 contrib usages inside TF2 runtime
        out = trigger_tf_contrib_deprecation()
        log(out, level="WARNING")
        return {"status": "tf_contrib_deprecation_triggered"}

    @app.get("/deprecation/sklearn-joblib")
    def dep_sklearn_joblib():
        # mhmd: prevent reliance on sklearn.externals.joblib shim
        out = trigger_sklearn_joblib_deprecation()
        log(out, level="WARNING")
        return {"status": "sklearn_joblib_deprecation_triggered"}
