"""
AI Embeddings and NLP Errors (e_ai_41 - e_ai_60)
Covers: Embeddings, text processing, sequence operations, attention mechanisms
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Embeddings and NLP Errors (41 - 60)
# ============================================================

def e_ai_41():
    """Embedding with out-of-vocabulary index"""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(50, 32, input_length=10),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    x = tf.random.uniform((1, 10), maxval=100, dtype=tf.int32)
    y = model(x)


def e_ai_42():
    """Random forest invalid max_depth"""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    model = RandomForestRegressor(max_depth="invalid")
    model.fit(X, y)


def e_ai_43():
    """Double backward error"""
    import torch
    x = torch.randn(5, 5, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    z = (y ** 2).sum()
    z.backward()


def e_ai_44():
    """TensorFlow tape already used"""
    import tensorflow as tf
    x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(x ** 2)
    grad = tape.gradient(y, x)
    grad = tape.gradient(y, x)


def e_ai_45():
    """Metric length mismatch"""
    import sklearn.metrics as metrics
    y_true = [0, 1, 2]
    y_pred = [0, 2]
    score = metrics.accuracy_score(y_true, y_pred)


def e_ai_46():
    """Packed sequence length incompatibility"""
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence
    x = torch.randn(10, 20)
    lengths = torch.LongTensor([25, 15, 20])
    packed = pack_padded_sequence(x, lengths, enforce_sorted=False)


def e_ai_47():
    """Model loss computation"""
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    x = tf.random.uniform((5, 5))
    y = tf.constant([0, 1, 2, 3, 4])
    model.fit(x, y, epochs=1)


def e_ai_48():
    """Transformer config with negative hidden size"""
    from transformers import AutoConfig, AutoModel
    import torch
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.hidden_size = -100
    model = AutoModel.from_config(config)
    model(torch.randn(1, 10, config.hidden_size))


def e_ai_49():
    """Batch norm incompatible dimensions"""
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.BatchNorm1d(10)
    )
    x = torch.randn(5, 10)
    y = model(x)


def e_ai_50():
    """PCA more components than samples"""
    import numpy as np
    from sklearn.decomposition import PCA
    X = np.random.randn(5, 100)
    pca = PCA(n_components=150)
    X_transformed = pca.fit_transform(X)


def e_ai_51():
    """Infinity in tensor"""
    import tensorflow as tf
    tf.debugging.assert_all_finite(tf.constant([1.0, float('inf')]), "Check failed")


def e_ai_52():
    """Index out of bounds"""
    import torch
    x = torch.randn(5, 5)
    indices = torch.LongTensor([10, 3, 1])
    y = x[indices]


def e_ai_53():
    """LabelEncoder unseen label"""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(['cat', 'dog', 'bird'])
    result = encoder.transform(['cat', 'elephant'])


def e_ai_54():
    """Attention with mismatched dimensions"""
    import torch.nn as nn
    import torch
    attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
    q = torch.randn(10, 2, 64)
    k = torch.randn(10, 2, 32)
    v = torch.randn(10, 2, 64)
    output = attention(q, k, v)


def e_ai_55():
    """Softmax invalid axis"""
    import tensorflow as tf
    x = tf.random.uniform((10, 20))
    y = tf.nn.softmax(x, axis=5)


def e_ai_56():
    """Adaptive pool size too large"""
    import torch
    from torch.nn import functional as F
    x = torch.randn(2, 3, 4, 5)
    y = F.adaptive_avg_pool2d(x, output_size=(10, 10))


def e_ai_57():
    """KNN n_neighbors too large"""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    X = np.random.randn(10, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X, y)
    knn.predict(np.random.randn(1, 5))


def e_ai_58():
    """Optimizer with empty parameters"""
    import torch
    import torch.optim as optim
    model = torch.nn.Linear(5, 2)
    optimizer = optim.SGD([])
    x = torch.randn(1, 5)
    loss = model(x).sum()


def e_ai_59():
    """TF function with variable"""
    import tensorflow as tf
    @tf.function
    def process(x):
        return x + tf.Variable([1.0])
    process(tf.constant([1.0, 2.0]))


def e_ai_60():
    """Duplicate weight norm"""
    import torch
    from torch.nn.utils import weight_norm
    layer = torch.nn.Linear(5, 5)
    layer = weight_norm(layer, name="weight")
    layer = weight_norm(layer, name="weight")


# Registry
__all__ = [f'e_ai_{i:02d}' for i in range(41, 61)]
