from fastapi import FastAPI
from utils.logger import log
import traceback
from error_propagation.controller import handle_request


def T(e):
    """Helper to log + raise consistently."""
    log(traceback.format_exc(), level="ERROR")
    raise e


# ================================================================
#     ORIGINAL ERRORS (RENAMED TO GENERIC names)
# ================================================================

def e_trig_01():
    import numpy as np
    try:
        # deprecated: np.asscalar
        x = np.array([5])
        np.asscalar(x)
    except Exception as e:
        T(e)


def e_trig_02():
    import numpy as np
    try:
        a = np.ones((3, 3))
        b = np.ones((4,))
        _ = a + b
    except Exception as e:
        T(e)


def e_trig_03():
    import numpy as np
    try:
        arr = np.array([1, 2, 3])
        _ = arr[10]
    except Exception as e:
        T(e)


def e_trig_04():
    import pandas as pd
    try:
        df = pd.DataFrame({"a": [1]})
        df = df.append({"a": 2})
    except Exception as e:
        T(e)


def e_trig_05():
    import pandas as pd
    try:
        df1 = pd.DataFrame({"id": [1], "name": ["A"]})
        df2 = pd.DataFrame({"user": [1], "age": [20]})
        pd.merge(df1, df2, on="id")
    except Exception as e:
        T(e)


def e_trig_06():
    import tensorflow as tf
    try:
        x = tf.ones((5,))
        _ = tf.compat.v1.nn.dropout(x, keep_prob=0.5)
    except Exception as e:
        T(e)


def e_trig_07():
    import tensorflow as tf
    try:
        a = tf.ones((3, 3))
        b = tf.ones((4, 4))
        tf.matmul(a, b)
    except Exception as e:
        T(e)


def e_trig_08():
    import torch
    try:
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(3, dtype=torch.float32)
        _ = a + b
    except Exception as e:
        T(e)


def e_trig_09():
    import torch
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.randn(5).cuda()
    except Exception as e:
        T(e)


def e_trig_10():
    from sklearn.linear_model import LinearRegression
    try:
        model = LinearRegression(normalize=True)
        model.fit([[1], [2], [3]], [1, 2, 3])
    except Exception as e:
        T(e)


def e_trig_11():
    from sklearn.linear_model import LogisticRegression
    try:
        model = LogisticRegression()
        model.fit([[1, 2], [3, 4]], [0])
    except Exception as e:
        T(e)


def e_trig_12():
    from sklearn.ensemble import RandomForestClassifier
    try:
        model = RandomForestClassifier(n_estimators="invalid")
        model.fit([[1], [2]], [0, 1])
    except Exception as e:
        T(e)


def e_trig_13():
    try:
        _ = 1 / 0
    except Exception as e:
        T(e)


def e_trig_14():
    try:
        x = None
        x.strip()
    except Exception as e:
        T(e)


def e_trig_15():
    try:
        d = {"a": 1}
        _ = d["b"]
    except Exception as e:
        T(e)


def e_trig_16():
    try:
        open("this_file_does_not_exist_abc123.txt", "r")
    except Exception as e:
        T(e)


def e_trig_17():
    try:
        import nonexistent_module
    except Exception as e:
        T(e)


def e_trig_18():
    try:
        _ = "a" + 5
    except Exception as e:
        T(e)


def e_trig_19():
    """Your real-world multi-file exception."""
    try:
        bad_request = {
            "user_id": 42,
            "payload": [5, 7]  # missing minimum size
        }
        handle_request(bad_request)
    except Exception as e:
        T(e)


# ================================================================
#       ADD 15 NEW REALISTIC PAGNDA/NUMPY/TF/TORCH/SKLEARN/PY ERRORS
# ================================================================

def e_trig_20():
    import numpy as np
    try:
        np.linalg.inv(np.zeros((3, 3)))
    except Exception as e:
        T(e)


def e_trig_21():
    import pandas as pd
    try:
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        df.astype(float)
    except Exception as e:
        T(e)


def e_trig_22():
    import numpy as np
    try:
        arr = np.arange(5)
        arr.reshape((2, 3))
    except Exception as e:
        T(e)


def e_trig_23():
    import tensorflow as tf
    try:
        a = tf.random.uniform((2, 5))
        b = tf.random.uniform((6, 2))
        tf.matmul(a, b)
    except Exception as e:
        T(e)


def e_trig_24():
    import torch
    from torch import nn
    try:
        m = nn.Linear(10, 5)
        m(torch.randn(3))  # wrong shape
    except Exception as e:
        T(e)


def e_trig_25():
    import sklearn
    from sklearn.svm import SVC
    try:
        model = SVC(kernel="unknown_kernel")
        model.fit([[1], [2]], [0, 1])
    except Exception as e:
        T(e)


def e_trig_26():
    import pandas as pd
    try:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.groupby("missing_col").sum()
    except Exception as e:
        T(e)


def e_trig_27():
    import numpy as np
    try:
        np.random.choice([], size=1)
    except Exception as e:
        T(e)


def e_trig_28():
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model("missing_model_dir")
    except Exception as e:
        T(e)


def e_trig_29():
    import torch
    try:
        x = torch.tensor([1.0], requires_grad=False)
        (x * 3).backward()
    except Exception as e:
        T(e)


def e_trig_30():
    import pandas as pd
    try:
        df = pd.DataFrame({"x": [1, 2, 3]})
        df["bad"] = df["x"].apply(lambda x: x["no"])
    except Exception as e:
        T(e)


def e_trig_31():
    try:
        import requests
        requests.get("http://127.0.0.1:99999")
    except Exception as e:
        T(e)


def e_trig_32():
    try:
        raise RuntimeError("Artificial runtime failure for pipeline step")
    except Exception as e:
        T(e)


def e_trig_33():
    import json
    try:
        json.loads("{ bad json }")
    except Exception as e:
        T(e)


def e_trig_34():
    import sqlite3
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("SELECT * FROM nonexistent_table")
    except Exception as e:
        T(e)


def e_trig_35():
    try:
        raise ValueError("Incorrect schema: missing field 'user_id'")
    except Exception as e:
        T(e)


# ================================================================
#             REGISTER + COMBINED ROUTES
# ================================================================

def register_routes(app: FastAPI):

    # keep your individual endpoints if you want debugging
    @app.get("/trigger/numpy")
    def a(): return e_trig_01()

    # ... skip repeating 20+ small API endpoints ...
    # You already know how to map them; you want a unified runner now.


    @app.get("/trigger/run-all")
    def run_all_triggers():
        """
        Run ALL exceptions sequentially to populate your dataset.
        Nothing stops execution except logging the exceptions.
        """
        funcs = [
            e_trig_01, e_trig_02, e_trig_03, e_trig_04, e_trig_05,
            e_trig_06, e_trig_07, e_trig_08, e_trig_09, e_trig_10,
            e_trig_11, e_trig_12, e_trig_13, e_trig_14, e_trig_15,
            e_trig_16, e_trig_17, e_trig_18, e_trig_19, e_trig_20,
            e_trig_21, e_trig_22, e_trig_23, e_trig_24, e_trig_25,
            e_trig_26, e_trig_27, e_trig_28, e_trig_29, e_trig_30,
            e_trig_31, e_trig_32, e_trig_33, e_trig_34, e_trig_35
        ]

        for fn in funcs:
            try:
                fn()
            except Exception:
                pass

        return {"status": "done", "executed": len(funcs)}
