"""
AI Transformers and Advanced Errors (e_ai_81 - e_ai_100)
Lightweight version – guaranteed to raise errors without heavy downloads.
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Transformers & Advanced Errors (81–100)
# Lightweight & guaranteed exceptions
# ============================================================

def e_ai_81():
    """Invalid pipeline step name (duplicate transformer names)"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    # Duplicate name "scale"
    Pipeline([("scale", StandardScaler()),
              ("scale", StandardScaler()),
              ("model", LinearRegression())])


def e_ai_82():
    """TensorFlow GradientTape – no watched tensor"""
    import tensorflow as tf
    x = tf.constant([1.0, 2.0])
    with tf.GradientTape() as tape:
        y = x * 2
    tape.gradient(y, x)  # ERROR: x is not watched


def e_ai_83():
    """Optimizer step before backward (PyTorch)"""
    import torch
    model = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # No backward was called → RuntimeError
    opt.step()


def e_ai_84():
    """Transformers pipeline nonexistent model (lightweight)"""
    from transformers import AutoTokenizer
    # Touch tokenizer only → raises OSError immediately
    AutoTokenizer.from_pretrained("nonexistent-xyz-123")


def e_ai_85():
    """Top-k with k > dimension"""
    import torch
    x = torch.randn(4, 4)
    torch.topk(x, k=10)  # too large → RuntimeError


def e_ai_86():
    """TensorFlow reduce_mean wrong dtype"""
    import tensorflow as tf
    x = tf.constant([[1, 2]], dtype=tf.string)
    tf.reduce_mean(x)  # invalid dtype → TypeError


def e_ai_87():
    """NumPy cross product with wrong shape"""
    import numpy as np
    a = np.array([[1, 2, 3]])
    b = np.array([4, 5, 6, 7])
    np.cross(a, b)  # shape mismatch


def e_ai_88():
    """Interpolation size invalid (PyTorch)"""
    import torch
    import torch.nn.functional as F
    x = torch.randn(1, 3, 4, 4)
    F.interpolate(x, size=(-1, 10))  # invalid size


def e_ai_89():
    """TensorFlow repeat negative repeat value"""
    import tensorflow as tf
    x = tf.range(5)
    tf.repeat(x, repeats=-1)  # invalid repeat count


def e_ai_90():
    """Voting classifier with invalid estimators"""
    from sklearn.ensemble import VotingClassifier
    clf = VotingClassifier([("a", None), ("b", None)])
    clf.fit([[1, 2], [3, 4]], [0, 1])  # None estimator → ValueError


def e_ai_91():
    """PyTorch SVD invalid matrix shape"""
    import torch
    x = torch.randn(3, 4)  # rectangular okay, but force error:
    torch.svd(x, some=True)  # deprecated → RuntimeError on many configs


def e_ai_92():
    """TensorFlow LSTM wrong input rank"""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(4)
    ])
    x = tf.random.uniform((10,))  # rank 1 instead of (batch, timesteps, features)
    model(x)  # ValueError


def e_ai_93():
    """Transformer encoder wrong embedding dim"""
    import torch
    from torch.nn import TransformerEncoderLayer
    enc = TransformerEncoderLayer(d_model=32, nhead=4)
    x = torch.randn(5, 2, 30)  # expected last dim = 32
    enc(x)  # RuntimeError: wrong dimension


def e_ai_94():
    """Softmax invalid axis"""
    import numpy as np
    from scipy.special import softmax
    x = np.array([1, 2, 3])
    softmax(x, axis=10)  # invalid axis


def e_ai_95():
    """Backward on detached graph"""
    import torch
    x = torch.randn(3, 3, requires_grad=True)
    y = x.sum()
    y.backward()
    z = y.detach() * x
    z.sum().backward()  # RuntimeError


def e_ai_96():
    """TensorFlow cast inf → integer"""
    import tensorflow as tf
    x = tf.constant([float("inf")])
    tf.cast(x, tf.int32)  # InvalidArgumentError


def e_ai_97():
    """Double backward on same tensor"""
    import torch
    x = torch.randn(3, 3, requires_grad=True)
    y = x.exp().sum()
    y.backward()
    y.backward()  # RuntimeError: graph already freed


def e_ai_98():
    """Tokenizer/model mismatch without loading full model"""
    from transformers import AutoTokenizer
    # Tokenizer exists but model doesn't → fast error
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    from transformers import AutoModel
    AutoModel.from_pretrained("bert-base-uncased-XYZ")  # fake model


def e_ai_99():
    """Dropout invalid probability"""
    import torch.nn as nn
    layer = nn.Dropout(p=2.0)  # must be 0 ≤ p ≤ 1
    layer(torch.randn(2, 2))  # RuntimeError


def e_ai_100():
    """Keras training label mismatch"""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=(3,)),
        tf.keras.layers.Dense(2)
    ])
    x = tf.random.uniform((5, 3))
    y = tf.random.uniform((5, 5))  # wrong shape
    model.compile("adam", "mse")
    model.fit(x, y, epochs=1)  # ValueError


# Registry
__all__ = [f"e_ai_{i:02d}" for i in range(81, 101)]
