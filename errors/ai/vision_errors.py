"""
AI Vision and CNN Errors (e_ai_21 - e_ai_40)
Covers: ResNet, Conv2D, dimension mismatches, device errors
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Vision and CNN Errors (21 - 40)
# ============================================================

def e_ai_21():
    """Linear layer dimension mismatch"""
    import torch
    layer = torch.nn.Linear(5, 5)
    input_tensor = torch.randn(10, 3)
    output = layer(input_tensor)


def e_ai_22():
    """CSV file not found"""
    import pandas as pd
    import os
    csv_path = "definitely_missing_file_abc123.csv"
    if not os.path.exists(csv_path):
        csv_path = "../" + csv_path
    df = pd.read_csv(csv_path)


def e_ai_23():
    """Reshape incompatible dimensions"""
    import numpy as np
    arr = np.arange(5)
    reshaped = arr.reshape((3, 3))


def e_ai_24():
    """CUDA device not available"""
    import torch
    try:
        tensor = torch.zeros((10, 10), device="cuda:50")
    except:
        tensor = torch.zeros((10, 10), device="invalid_device")


def e_ai_25():
    """Imputer on mixed types"""
    import pandas as pd
    from sklearn.impute import SimpleImputer
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    imputer = SimpleImputer(strategy="mean")
    result = imputer.fit_transform(df)


def e_ai_26():
    """LSTM input shape mismatch"""
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.LSTM(4)])
    input_data = tf.random.uniform((1, 5))
    output = model(input_data)


def e_ai_27():
    """Tokenizer missing pad token"""
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.pad_token = None
    tokens = tokenizer(["a", "b"], padding=True)


def e_ai_28():
    """Backward on detached tensor"""
    import torch
    x = torch.randn((5, 5), requires_grad=True)
    y = x.detach()
    y.backward(torch.ones_like(y))


def e_ai_29():
    """Random choice from empty array"""
    import numpy as np
    empty_array = np.array([])
    choice = np.random.choice(empty_array)


def e_ai_30():
    """Decision tree invalid max_depth"""
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth="not_a_number")
    clf.fit([[1, 2], [3, 4]], [0, 1])


def e_ai_31():
    """TensorFlow model not found"""
    import tensorflow as tf
    import os
    model_dir = "missing_model_dir/"
    if not os.path.exists(model_dir):
        model_dir = model_dir.replace("missing", "not_found")
    model = tf.keras.models.load_model(model_dir)


def e_ai_32():
    """Memory error - huge tensor"""
    import torch
    huge_tensor = torch.randn(int(1e15), int(1e15))
    torch.save(huge_tensor, "bigfile.pt")


def e_ai_33():
    """Unknown model revision"""
    import transformers
    model = transformers.AutoModel.from_pretrained(
        "bert-base-uncased",
        revision="unknown_tag_12345"
    )


def e_ai_34():
    """Invalid dictionary indexing"""
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3]})
    df["bad"] = df["x"].apply(lambda v: v["bad"])


def e_ai_35():
    """Null pointer dereference"""
    pipeline_output = None
    if pipeline_output is None:
        processed = 1 / len(pipeline_output)


def e_ai_36():
    """ResNet channel mismatch"""
    import torch
    import torchvision
    image = torch.randn(1, 4, 224, 224)
    model = torchvision.models.resnet18(pretrained=False)
    output = model(image)


def e_ai_37():
    """Conv2D applied to wrong tensor"""
    import torch.nn as nn
    import torch
    layer = nn.Linear(10, 5)
    x = torch.randn(5, 10)
    y = layer(x)
    z = nn.Conv2d(5, 16, 3)(y)


def e_ai_38():
    """Autoencoder shape mismatch"""
    import tensorflow as tf
    encoder = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(20,))])
    decoder = tf.keras.Sequential([tf.keras.layers.Dense(20)])
    x = tf.random.uniform((1, 15))
    z = encoder(x)
    y = decoder(z)


def e_ai_39():
    """Negative learning rate"""
    import torch
    from torch.optim import Adam
    model = torch.nn.Linear(5, 2)
    optimizer = Adam(model.parameters(), lr=-0.001)
    x = torch.randn(1, 5)
    loss = model(x).sum()


def e_ai_40():
    """Embedding index out of bounds"""
    import torch
    import torch.nn as nn
    embedding_layer = nn.Embedding(100, 50)
    indices = torch.LongTensor([0, 105, 50])
    embeddings = embedding_layer(indices)


# Registry
__all__ = [f'e_ai_{i:02d}' for i in range(21, 41)]
