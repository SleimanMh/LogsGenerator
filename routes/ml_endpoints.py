from fastapi import FastAPI
from utils.logger import log
from utils.traceback_utils import traceback_block


def register_ml_routes(app: FastAPI):

    # -----------------------------
    #  EXISTING 15 ML ERRORS
    # -----------------------------

    def e_shape_mismatch():
        try:
            import torch
            A = torch.rand((3, 4))
            B = torch.rand((5,))
            _ = A @ B
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_sklearn_length_mismatch():
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import load_iris
            X, y = load_iris(return_X_y=True)
            LogisticRegression().fit(X, y[:-5])
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_input_mismatch():
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(5,))])
            x = tf.random.uniform((1, 1))
            model(x)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_sklearn_not_fitted():
        try:
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            scaler = StandardScaler()
            scaler.transform(np.random.rand(8, 3))
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_state_mismatch():
        try:
            import torch
            from torch import nn
            layer = nn.Linear(4, 2)
            bogus = {"weight": torch.rand((3, 3)), "bias": torch.rand(4)}
            layer.load_state_dict(bogus)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_device():
        try:
            import torch
            torch.zeros((16, 16), device="cuda:99")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_load_fail():
        try:
            import tensorflow as tf
            tf.saved_model.load("non-existent-export")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_oom():
        import torch
        try:
            torch.empty((10**9, 10**9), device="cuda")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_grad():
        import torch
        try:
            x = torch.tensor([1.0], requires_grad=False)
            y = x * 2
            y.backward()
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_oom():
        import tensorflow as tf
        try:
            x = tf.random.uniform((10_000, 10_000, 1000))
            tf.matmul(x, x)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_model_load_fail():
        import torch
        try:
            torch.load("non_existent_checkpoint.pt")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_onnx_runtime():
        try:
            import onnxruntime as ort
            ort.InferenceSession("non_existent.onnx")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tokenizer_error():
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained("nonexistent-model-123")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_hf_weight_mismatch():
        try:
            from transformers import BertModel
            model = BertModel.from_pretrained("bert-base-uncased")
            model.load_state_dict({"unexpected": 42})
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_data_shape_mismatch():
        import numpy as np
        try:
            X = np.random.rand(100, 5)
            y = np.random.rand(99)
            from sklearn.linear_model import LinearRegression
            LinearRegression().fit(X, y)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_nccl():
        try:
            raise RuntimeError("NCCL error: unhandled system error")
        except:
            log(traceback_block(), level="ERROR")
            raise


    # -----------------------------
    #  NEW 20 ML ERRORS
    # -----------------------------

    def e_tensor_type():
        try:
            import torch
            a = torch.randn((2, 2), dtype=torch.float16)
            b = torch.randn((2, 2), dtype=torch.float64)
            _ = a + b
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_invalid_lr_param():
        try:
            from sklearn.linear_model import LogisticRegression
            LogisticRegression(C="invalid")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_pytorch_size_mismatch():
        try:
            import torch
            m = torch.nn.Linear(10, 5)
            x = torch.randn((1, 8))
            m(x)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_wrong_loss():
        try:
            import tensorflow as tf
            loss = tf.keras.losses.BinaryCrossentropy()
            loss([0.1, 0.2], [1.0, 0.0, 0.5])
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_sklearn_invalid_solver():
        try:
            from sklearn.linear_model import LogisticRegression
            LogisticRegression(solver="not_real").fit([[1]], [0])
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_np_broadcast_fail():
        import numpy as np
        try:
            a = np.ones((5, 5))
            b = np.ones((3,))
            _ = a + b
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_layer_fail():
        try:
            import tensorflow as tf
            x = tf.random.uniform((2, 10))
            layer = tf.keras.layers.Dense(5)
            y = layer(x)
            z = tf.matmul(y, tf.ones((3, 3)))
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_hf_tokenizer_call_fail():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            tok(None)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_backward_twice():
        import torch
        try:
            x = torch.tensor(3.0, requires_grad=True)
            y = x * 5
            y.backward()
            y.backward()
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_no_grad():
        try:
            import tensorflow as tf
            x = tf.constant(3.0)
            with tf.GradientTape() as tape:
                y = x * 2
            tape.gradient(y, x)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_onnx_missing_input():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession("non_existent.onnx")
            sess.run(None, {"missing_input": [1, 2]})
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_invalid_dim():
        import torch
        try:
            x = torch.randn((2, 3, 4))
            torch.nn.Flatten()(x, start_dim=5)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_np_lin_alg():
        import numpy as np
        try:
            np.linalg.inv(np.zeros((3, 3)))
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_model_predict_fail():
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Dense(3)])
            model.predict(np.ones((1,)))  # wrong shape
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_sklearn_predict_before_fit():
        try:
            from sklearn.linear_model import LinearRegression
            LinearRegression().predict([[1, 2]])
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_invalid_activation():
        import torch.nn as nn
        try:
            nn.ReLU()(None)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_tf_wrong_dtype():
        try:
            import tensorflow as tf
            x = tf.constant(["a", "b"])
            tf.math.reduce_mean(x)
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_conv_fail():
        try:
            import torch.nn as nn
            import torch
            conv = nn.Conv2d(3, 6, 3)
            conv(torch.randn((1, 1, 32, 32)))  # wrong channels
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_torch_missing_cuda():
        try:
            import torch
            torch.tensor([1]).to("cuda:50")
        except:
            log(traceback_block(), level="ERROR")
            raise

    def e_pipeline_custom_fail():
        try:
            raise ValueError("Pipeline step failed: invalid intermediate tensor")
        except:
            log(traceback_block(), level="ERROR")
            raise



    # ======================================================
    #   ONE MAIN ROUTE THAT TRIGGERS ALL 35 EXCEPTIONS
    # ======================================================
    @app.get("/ml/run-all")
    def ml_run_all():
        functions = [
            e_shape_mismatch,
            e_sklearn_length_mismatch,
            e_tf_input_mismatch,
            e_sklearn_not_fitted,
            e_torch_state_mismatch,
            e_torch_device,
            e_tf_load_fail,
            e_torch_oom,
            e_torch_grad,
            e_tf_oom,
            e_model_load_fail,
            e_onnx_runtime,
            e_tokenizer_error,
            e_hf_weight_mismatch,
            e_data_shape_mismatch,
            e_nccl,

            # New 20
            e_tensor_type,
            e_invalid_lr_param,
            e_pytorch_size_mismatch,
            e_tf_wrong_loss,
            e_sklearn_invalid_solver,
            e_np_broadcast_fail,
            e_tf_layer_fail,
            e_hf_tokenizer_call_fail,
            e_torch_backward_twice,
            e_tf_no_grad,
            e_onnx_missing_input,
            e_torch_invalid_dim,
            e_np_lin_alg,
            e_tf_model_predict_fail,
            e_sklearn_predict_before_fit,
            e_torch_invalid_activation,
            e_tf_wrong_dtype,
            e_torch_conv_fail,
            e_torch_missing_cuda,
            e_pipeline_custom_fail,
        ]

        for fn in functions:
            try:
                fn()
            except Exception:
                pass  # let pipeline capture errors but continue

        return {"status": "done", "executed": len(functions)}
