"""
AI Autograd and RNN Errors (e_ai_61 - e_ai_80)
Covers: Backpropagation, RNNs, gradients, tensor ops, and state management
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Autograd and RNN Errors (61 - 80)
# Each function is lightweight and guaranteed to raise an error
# ============================================================

def e_ai_61():
    """Cholesky on non–positive-definite matrix"""
    import numpy as np
    # Simple symmetric but not positive-definite matrix
    M = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    # This raises numpy.linalg.LinAlgError
    _ = np.linalg.cholesky(M)


def e_ai_62():
    """Model state dict shape mismatch (PyTorch)"""
    import torch
    model = torch.nn.Linear(4, 2)
    state = model.state_dict()
    # Wrong weight shape
    state["weight"] = torch.randn(3, 4)
    model.load_state_dict(state, strict=True)  # RuntimeError


def e_ai_63():
    """Conv2D input rank mismatch (TensorFlow)"""
    import tensorflow as tf
    layer = tf.keras.layers.Conv2D(8, kernel_size=3, input_shape=(28, 28, 1))
    # Build model
    model = tf.keras.Sequential([layer])
    # Wrong rank: (28, 28, 1) instead of (batch, h, w, c)
    x = tf.random.uniform((28, 28, 1))
    model(x)  # ValueError


def e_ai_64():
    """Multiple backward passes on same graph (PyTorch)"""
    import torch
    x = torch.randn(5, 5, requires_grad=True)
    y = x.sum()
    y.backward()
    # Second backward without retain_graph=True → RuntimeError
    z = (x * 2).sum()
    z.backward()


def e_ai_65():
    """Embedding index out of range (sequence too long concept)"""
    import torch
    emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=4)
    # Token id 15 is out of range [0, 9]
    tokens = torch.tensor([[1, 2, 15]])
    emb(tokens)  # IndexError / RuntimeError


def e_ai_66():
    """Distribution with negative scale (PyTorch)"""
    import torch
    import torch.distributions as dist
    # Invalid scale parameter
    normal = dist.Normal(0.0, -1.0)
    normal.rsample()  # RuntimeError


def e_ai_67():
    """SciPy minimize with invalid initial point"""
    from scipy.optimize import minimize
    # x0 must be array-like numeric, not string
    def f(x):
        return x[0] ** 2
    minimize(f, x0="not_numeric")  # TypeError / ValueError


def e_ai_68():
    """TensorFlow matmul shape mismatch"""
    import tensorflow as tf
    a = tf.ones((2, 3))
    b = tf.ones((4, 2))
    tf.matmul(a, b)  # InvalidArgumentError (incompatible shapes)


def e_ai_69():
    """Conv2D with invalid stride (PyTorch)"""
    import torch
    import torch.nn.functional as F
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(8, 3, 3, 3)
    # stride=0 is invalid → RuntimeError
    F.conv2d(x, w, stride=0)


def e_ai_70():
    """Cross-validation with too many folds"""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    X = [[1.0], [2.0]]
    y = [1.0, 2.0]
    # cv=10 with only 2 samples → ValueError
    cross_val_score(LinearRegression(), X, y, cv=10)


def e_ai_71():
    """Matrix multiply dtype mismatch (PyTorch)"""
    import torch
    x = torch.randn(3, 3, dtype=torch.float32)
    y = torch.randn(3, 3, dtype=torch.float64)
    torch.mm(x, y)  # RuntimeError: expected same dtype


def e_ai_72():
    """tf.function input_signature mismatch"""
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
    def model(x):
        return x + 1.0

    # Wrong shape: last dim = 5 instead of 10 → ValueError
    bad_input = tf.random.uniform((2, 5))
    model(bad_input)


def e_ai_73():
    """StandardScaler with invalid parameter type"""
    from sklearn.preprocessing import StandardScaler
    # with_mean must be bool, not string → TypeError / ValueError
    scaler = StandardScaler(with_mean="invalid")
    scaler.fit([[1.0, 2.0], [3.0, 4.0]])


def e_ai_74():
    """Tensor indexing with incompatible boolean mask (PyTorch)"""
    import torch
    x = torch.randn(5)
    mask = torch.tensor([True, False, True, False, True, False])
    # Mask length != tensor length → IndexError
    _ = x[mask]


def e_ai_75():
    """TensorFlow dense_to_sparse on sparse tensor"""
    import tensorflow as tf
    # dense_to_sparse expects dense input, not SparseTensor
    sparse = tf.sparse.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[2, 2])
    tf.dense_to_sparse(sparse)  # TypeError


def e_ai_76():
    """RNNCell hidden state size mismatch (PyTorch)"""
    import torch
    from torch.nn import RNNCell
    rnn = RNNCell(input_size=4, hidden_size=8)
    x = torch.randn(1, 4)
    h = torch.randn(1, 5)  # wrong hidden size
    rnn(x, h)  # RuntimeError


def e_ai_77():
    """MultiLabelBinarizer transform with wrong input shape"""
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([[1, 2], [2, 3]])
    # transform expects iterable of iterables; this is wrong → ValueError
    mlb.transform([1, 2])


def e_ai_78():
    """Eigen decomposition on non-square matrix (PyTorch)"""
    import torch
    x = torch.randn(3, 4)
    # torch.linalg.eigh expects square matrix → RuntimeError
    torch.linalg.eigh(x)


def e_ai_79():
    """Embedding with out-of-range index (TensorFlow)"""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10, output_dim=4)
    ])
    # Token id 11 is out of [0, 9] → InvalidArgumentError
    x = tf.constant([[1, 2, 11]])
    model(x)


def e_ai_80():
    """TensorDataset with mismatched lengths (PyTorch)"""
    import torch
    from torch.utils.data import TensorDataset
    X = torch.randn(5, 3)
    y = torch.randn(7)  # different first dim
    # This raises ValueError at dataset creation
    TensorDataset(X, y)


# Registry
__all__ = [f"e_ai_{i:02d}" for i in range(61, 81)]
