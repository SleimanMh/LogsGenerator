"""
AI Data Processing & Preprocessing Errors (e_ai_01 - e_ai_20)
Covers: Data loading, preprocessing, version mismatches, type errors
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Data Processing & Preprocessing Errors (01 - 20)
# ============================================================

def e_ai_01():
    """Data preprocessing type mismatch"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    df = pd.DataFrame({"age": [25, 32, None], "city": ["Beirut", "Paris", "Berlin"]})
    df["age"] = df["age"].fillna(999)
    df["age"] = df["age"].astype(str)
    scaler = StandardScaler()
    features = df[["age"]].values
    scaled = scaler.fit_transform(features)


def e_ai_02():
    """Missing required columns"""
    import pandas as pd
    expected_cols = ["feature1", "feature2", "feature3"]
    incoming = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    selected = incoming[expected_cols]


def e_ai_03():
    """String data to numeric model"""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    X = np.array([["one", "two"], ["three", "four"]])
    y = np.array([1.0, 2.0])
    model = LinearRegression()
    model.fit(X, y)


def e_ai_04():
    """Feature dimension mismatch"""
    import numpy as np
    expected_features = 12
    raw_input = np.random.randn(5, 10)
    model_state = {"n_features": expected_features}
    if raw_input.shape[1] != model_state["n_features"]:
        raw_input = raw_input[:, :expected_features]
    result = raw_input[:, :expected_features]


def e_ai_05():
    """PyTorch version incompatibility"""
    import torch
    required_version = "2.1.0"
    current = torch.__version__
    version_tuple = tuple(map(int, current.split(".")[:2]))
    required_tuple = tuple(map(int, required_version.split(".")[:2]))
    if version_tuple < required_tuple:
        _ = 1 / 0


def e_ai_06():
    """Data drift detection failure"""
    import pandas as pd
    from scipy.stats import ks_2samp
    baseline = pd.Series([100, 150, 200, 250, 300])
    current = pd.Series([101, 105, 110, 115, 120])
    stat, pval = ks_2samp(baseline, current)
    if pval < 0.05:
        result = pval / 0


def e_ai_07():
    """Neural network shape mismatch"""
    import torch
    from torch import nn
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    x = torch.randn(1, 128)
    output = model(x)


def e_ai_08():
    """TensorFlow input shape error"""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(10,)),
        tf.keras.layers.Dense(4)
    ])
    batch = tf.random.uniform((1, 8))
    result = model(batch)


def e_ai_09():
    """Model state dict shape incompatibility"""
    import torch
    from torch import nn
    model = nn.Linear(10, 2)
    bogus_state = {"weight": torch.rand((3, 3)), "bias": torch.rand(4)}
    model.load_state_dict(bogus_state, strict=False)
    out = model(torch.randn(1, 10))


def e_ai_10():
    """Backpropagation on non-leaf tensor"""
    import torch
    x = torch.tensor([1.0], requires_grad=False)
    y = x * 2
    z = y.sum()
    z.backward()


def e_ai_11():
    """NaN loss propagation"""
    import torch
    from torch import nn
    pred = torch.tensor([float("nan")])
    target = torch.tensor([1.0])
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, target)
    if torch.isnan(loss):
        _ = loss.item() + (1 / (1 - loss.item()))


def e_ai_12():
    """Missing checkpoint file"""
    import torch
    import os
    checkpoint_path = "non_existent_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path[0] = "x"


def e_ai_13():
    """Tokenizer model not found"""
    from transformers import AutoTokenizer
    model_name = "nonexistent-model-123"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def e_ai_14():
    """BERT state dict strict loading error"""
    from transformers import BertModel
    model = BertModel.from_pretrained("bert-base-uncased")
    wrong_state = {"unexpected_key": 42}
    model.load_state_dict(wrong_state, strict=True)


def e_ai_15():
    """Classification model wrong label format"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model(**inputs, labels=torch.tensor([[1, 0]]))


def e_ai_16():
    """OpenAI API invalid parameters"""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=-5
    )


def e_ai_17():
    """Singular matrix inversion"""
    import numpy as np
    matrix = np.zeros((3, 3))
    inv = np.linalg.inv(matrix)


def e_ai_18():
    """Scaler on non-numeric data"""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    scaler = MinMaxScaler()
    result = scaler.fit_transform(df)


def e_ai_19():
    """Matrix multiplication dimension error"""
    import tensorflow as tf
    x = tf.random.uniform((2, 3))
    y = tf.random.uniform((4, 5))
    z = tf.matmul(x, y)


def e_ai_20():
    """Missing pickle file"""
    import joblib
    import os
    model_path = "missing_model.pkl"
    if not os.path.exists(model_path):
        model_path = model_path.replace("missing", "not_found")
    loaded = joblib.load(model_path)


# Registry
__all__ = [f'e_ai_{i:02d}' for i in range(1, 21)]
