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


    @app.get("/ml/tensorflow-load")
    def ml_tf_load():
        # mhmd: ensure serving layers surface missing SavedModel exports
        try:
            import tensorflow as tf
            tf.saved_model.load("non-existent-export")
        except:
            log(traceback_block(), level="ERROR")
            raise
