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