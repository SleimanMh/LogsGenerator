# """
# AI Error Generator Project
# --------------------------
# A single Python script that generates rich, realistic machine-learning and AI-related errors
# from multiple libraries: PyTorch, TensorFlow, scikit-learn, Transformers, and pure Python.

# Running this script will:
#  - Execute multiple ML operations designed to fail
#  - Produce stack traces, warnings, deprecation errors, shape mismatches, file-loading errors, etc.
#  - Log everything to ./logs/ai_errors.log

# Usage:
#     python ai_error_generator.py

# Ensure you install: torch, tensorflow, scikit-learn, transformers
# (Use CPU versions; GPU optional)

#     pip install torch tensorflow scikit-learn transformers

# """

# import os
# import logging
# from datetime import datetime

# # Create log directory
# os.makedirs("logs", exist_ok=True)

# # Configure logging
# log_file = os.path.join("logs", "ai_errors.log")
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )

# print(f"Logging errors to: {log_file}")
# logging.info("==== AI ERROR GENERATION STARTED ====")

# #############################################
# # 1. PyTorch Errors
# #############################################
# def torch_errors():
#     logging.info("-- Running PyTorch error scenarios --")
#     try:
#         import torch
#         import torch.nn as nn

#         # Shape mismatch
#         A = torch.rand((3, 4))
#         B = torch.rand((5,))
#         C = A @ B  # will fail
#     except Exception:
#         logging.exception("PyTorch shape mismatch error")

#     try:
#         # CUDA error
#         import torch
#         torch.zeros((1000, 1000)).cuda()  # fails if no GPU
#     except Exception:
#         logging.exception("PyTorch CUDA failure")


# #############################################
# # 2. TensorFlow Errors
# #############################################
# def tensorflow_errors():
#     logging.info("-- Running TensorFlow error scenarios --")
#     try:
#         import tensorflow as tf

#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(10, input_shape=(5,))
#         ])
#         x = tf.random.uniform((1, 1))  # wrong shape
#         model(x)
#     except Exception:
#         logging.exception("TensorFlow shape mismatch error")

#     try:
#         # File load error
#         import tensorflow as tf
#         tf.keras.models.load_model("missing_model.h5")
#     except Exception:
#         logging.exception("TensorFlow model loading error")


# #############################################
# # 3. scikit-learn Errors
# #############################################
# def sklearn_errors():
#     logging.info("-- Running sklearn error scenarios --")
#     try:
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.datasets import load_iris
#         import numpy as np

#         X, y = load_iris(return_X_y=True)
#         model = LogisticRegression(max_iter=10)
#         model.fit(X, y[:-1])  # mismatched labels
#     except Exception:
#         logging.exception("sklearn label-length mismatch error")

#     try:
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.datasets import load_iris
#         import numpy as np

#         X, y = load_iris(return_X_y=True)
#         X[0, 0] = float('nan')
#         model = LogisticRegression(max_iter=10)
#         model.fit(X, y)
#     except Exception:
#         logging.exception("sklearn NaN input error")


# #############################################
# # 4. HuggingFace Transformers Errors
# #############################################
# def hf_errors():
#     logging.info("-- Running Transformers error scenarios --")
#     try:
#         from transformers import AutoTokenizer
#         AutoTokenizer.from_pretrained("non-existing-hf-model-xyz")
#     except Exception:
#         logging.exception("Transformers missing model/tokenizer error")

#     try:
#         from transformers import AutoModel
#         model = AutoModel.from_pretrained("bert-base-uncased")

#         # Wrong shape for forward pass
#         import torch
#         dummy = torch.zeros((3, 10))  # invalid for BERT
#         model(dummy)
#     except Exception:
#         logging.exception("Transformers forward-pass tensor shape error")


# #############################################
# # 5. Simple Python ML-like Errors
# #############################################
# def simple_python_errors():
#     logging.info("-- Running basic Python ML-like errors --")
#     try:
#         data = []
#         mean = sum(data) / len(data)  # ZeroDivisionError
#     except Exception:
#         logging.exception("ZeroDivisionError while computing mean")

#     try:
#         import json
#         json.loads("{bad json}")
#     except Exception:
#         logging.exception("JSON parsing error")

#     try:
#         import numpy as np
#         arr = np.arange(10)
#         print(arr[20])  # IndexError
#     except Exception:
#         logging.exception("Numpy index out-of-bounds error")


# #############################################
# # EXECUTE ALL ERROR GENERATORS
# #############################################
# if __name__ == "__main__":
#     torch_errors()
#     tensorflow_errors()
#     sklearn_errors()
#     hf_errors()
#     simple_python_errors()

#     logging.info("==== AI ERROR GENERATION COMPLETED ====")
#     print("All errors generated. Check logs/ai_errors.log")
