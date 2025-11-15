from fastapi import FastAPI
from utils.logger import log
from utils.traceback_utils import traceback_block

def register_ml_routes(app: FastAPI):
    @app.get("/ml/shape_mismatch")
    def ml_shape():
        try:
            import torch
            A = torch.rand((3, 4))
            B = torch.rand((5,))
            C = A @ B
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/ml/sklearn")
    def ml_sklearn():
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import load_iris
            X, y = load_iris(return_X_y=True)
            model = LogisticRegression()
            model.fit(X, y[:-5]) # length mismatch
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/ml/tensorflow")
    def ml_tf():
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(5,))
            ])
            x = tf.random.uniform((1, 1))
            model(x)
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/ml/sklearn-not-fitted")
    def ml_sklearn_not_fitted():
        # mhmd: catch pipelines calling transform/predict before fit
        try:
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            scaler = StandardScaler()
            data = np.random.rand(8, 3)
            scaler.transform(data)
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/ml/torch-state-mismatch")
    def ml_torch_state():
        # mhmd: detect incompatible model checkpoints during rollout
        try:
            import torch
            from torch import nn
            layer = nn.Linear(4, 2)
            bogus_state = {"weight": torch.rand((3, 3)), "bias": torch.rand(4)}
            layer.load_state_dict(bogus_state)
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/ml/torch-device")
    def ml_torch_device():
        # mhmd: verify deployments aren't hardcoding unavailable accelerators
        try:
            import torch
            torch.zeros((16, 16), device="cuda:99")
        except:
            log(traceback_block(), level="ERROR")
            raise


    @app.get("/ml/tensorflow-lo ad")
    def ml_tf_load():
        # mhmd: ensure serving layers surface missing SavedModel exports
        try:
            import tensorflow as tf
            tf.saved_model.load("non-existent-export")
        except:
            log(traceback_block(), level="ERROR")
            raise

    @app.get("/ml/torch-cuda-oom")
    def ml_torch_oom():
        import torch
        try:
            # simulate out-of-memory
            torch.empty((10**9, 10**9), device="cuda")
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/torch-gradient")
    def ml_torch_grad():
        import torch
        try:
            x = torch.tensor([1.0], requires_grad=False)
            y = x * 2
            y.backward()  # RuntimeError: element 0 does not require grad
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/tensorflow-oom")
    def ml_tf_oom():
        import tensorflow as tf
        try:
            x = tf.random.uniform((10_000, 10_000, 1000))
            y = tf.matmul(x, x)
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/model-load-fail")
    def ml_model_load_fail():
        import torch
        try:
            torch.load("non_existent_checkpoint.pt")
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/onnx-runtime-error")
    def ml_onnx_runtime():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession("non_existent_model.onnx")
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/tokenizer-error")
    def ml_tokenizer_error():
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("nonexistent-model-123")
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/huggingface-weight-mismatch")
    def ml_hf_weight_mismatch():
        from transformers import BertModel
        try:
            model = BertModel.from_pretrained("bert-base-uncased")
            wrong_state = {"unexpected_key": 42}
            model.load_state_dict(wrong_state)
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/data-shape-mismatch")
    def ml_data_shape_mismatch():
        import numpy as np
        try:
            X = np.random.rand(100, 5)
            y = np.random.rand(99)  # off by one
            from sklearn.linear_model import LinearRegression
            LinearRegression().fit(X, y)
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise


    @app.get("/ml/distributed-nccl-error")
    def ml_nccl_error():
        try:
            raise RuntimeError("NCCL error in: ../torch/lib/c10d/ProcessGroupNCCL.cpp:912, unhandled system error")
        except Exception as e:
            log(str(e).replace("Traceback (most recent call last):", ""), level="ERROR")
            raise
