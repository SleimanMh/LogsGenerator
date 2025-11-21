from fastapi import FastAPI
from utils.logger import log
import traceback
from routes.ai_pipeline import AIPipeline


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


def register_ai_routes(app: FastAPI):

    # ------------------------------------------------------
    #  EXISTING AI ERRORS (renamed for uniformity)
    # ------------------------------------------------------

    def e_ai_01():
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        df = pd.DataFrame({"age": [25, 32, None], "city": ["Beirut", "Paris", "Berlin"]})
        df["age"] = df["age"].fillna(999)
        df["age"] = df["age"].astype(str)
        scaler = StandardScaler()
        features = df[["age"]].values
        scaled = scaler.fit_transform(features)  # fails: cannot scale strings

    def e_ai_02():
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder
        expected_cols = ["feature1", "feature2", "feature3"]
        incoming = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        selected = incoming[expected_cols]  # fails: missing column

    def e_ai_03():
        import numpy as np
        from sklearn.linear_model import LinearRegression
        X = np.array([["one", "two"], ["three", "four"]])
        y = np.array([1.0, 2.0])
        model = LinearRegression()
        model.fit(X, y)  # fails: cannot fit string data

    def e_ai_04():
        import pickle
        import numpy as np
        expected_features = 12
        raw_input = np.random.randn(5, 10)
        model_state = {"n_features": expected_features}
        if raw_input.shape[1] != model_state["n_features"]:
            raw_input = raw_input[:, :expected_features]
        # implicit exception when slicing fails
        result = raw_input[:, :expected_features]

    def e_ai_05():
        import torch
        import importlib
        required_version = "2.1.0"
        current = torch.__version__
        version_tuple = tuple(map(int, current.split(".")[:2]))
        required_tuple = tuple(map(int, required_version.split(".")[:2]))
        if version_tuple < required_tuple:
            _ = 1 / 0  # implicit exception

    def e_ai_06():
        import pandas as pd
        from scipy.stats import ks_2samp
        baseline = pd.Series([100, 150, 200, 250, 300])
        current = pd.Series([101, 105, 110, 115, 120])
        stat, pval = ks_2samp(baseline, current)
        if pval < 0.05:
            result = pval / 0  # implicit exception on drift detection

    def e_ai_07():
        import torch
        from torch import nn
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        x = torch.randn(1, 128)
        output = model(x)  # fails: dimension mismatch 64 -> 32

    def e_ai_08():
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, input_shape=(10,)),
            tf.keras.layers.Dense(4)
        ])
        batch = tf.random.uniform((1, 8))
        result = model(batch)  # fails: input shape mismatch (8 vs 10)

    def e_ai_09():
        import torch
        from torch import nn
        model = nn.Linear(10, 2)
        bogus_state = {"weight": torch.rand((3, 3)), "bias": torch.rand(4)}
        model.load_state_dict(bogus_state, strict=False)  # fails: incompatible shapes
        out = model(torch.randn(1, 10))

    def e_ai_10():
        import torch
        x = torch.tensor([1.0], requires_grad=False)
        y = x * 2
        z = y.sum()
        z.backward()  # fails: requires_grad is False

    def e_ai_11():
        import torch
        from torch import nn
        pred = torch.tensor([float("nan")])
        target = torch.tensor([1.0])
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, target)
        if torch.isnan(loss):
            _ = loss.item() + (1 / (1 - loss.item()))  # implicit exception

    def e_ai_12():
        import torch
        import os
        checkpoint_path = "non_existent_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path[0] = "x"  # implicit: immutable string

    def e_ai_13():
        from transformers import AutoTokenizer
        model_name = "nonexistent-model-123"
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # implicit exception

    def e_ai_14():
        from transformers import BertModel
        model = BertModel.from_pretrained("bert-base-uncased")
        wrong_state = {"unexpected_key": 42}
        model.load_state_dict(wrong_state, strict=True)  # implicit exception

    def e_ai_15():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model(**inputs, labels=torch.tensor([[1, 0]]))  # implicit: wrong labels shape

    def e_ai_16():
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=-5  # implicit: negative parameter
        )

    # ------------------------------------------------------
    #   ADD 20 NEW AI/ML REALISTIC ERRORS
    # ------------------------------------------------------

    def e_ai_17():
        import numpy as np
        matrix = np.zeros((3, 3))
        inv = np.linalg.inv(matrix)  # implicit: singular matrix

    def e_ai_18():
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        scaler = MinMaxScaler()
        result = scaler.fit_transform(df)  # implicit: cannot scale strings

    def e_ai_19():
        import tensorflow as tf
        x = tf.random.uniform((2, 3))
        y = tf.random.uniform((4, 5))
        z = tf.matmul(x, y)  # implicit: incompatible dimensions

    def e_ai_20():
        import joblib
        import os
        model_path = "missing_model.pkl"
        if not os.path.exists(model_path):
            model_path = model_path.replace("missing", "not_found")
        loaded = joblib.load(model_path)  # implicit: file not found

    def e_ai_21():
        import torch
        layer = torch.nn.Linear(5, 5)
        input_tensor = torch.randn(10, 3)
        output = layer(input_tensor)  # implicit: wrong input dim (3 vs 5)

    def e_ai_22():
        import pandas as pd
        import os
        csv_path = "definitely_missing_file_abc123.csv"
        if not os.path.exists(csv_path):
            csv_path = "../" + csv_path
        df = pd.read_csv(csv_path)  # implicit: file not found

    def e_ai_23():
        import numpy as np
        arr = np.arange(5)
        reshaped = arr.reshape((3, 3))  # implicit: cannot reshape 5 elements to (3,3)

    def e_ai_24():
        import torch
        try:
            tensor = torch.zeros((10, 10), device="cuda:50")
        except:
            # fallback that also fails
            tensor = torch.zeros((10, 10), device="invalid_device")

    def e_ai_25():
        import pandas as pd
        from sklearn.impute import SimpleImputer
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        imputer = SimpleImputer(strategy="mean")
        result = imputer.fit_transform(df)  # implicit: mean strategy on mixed types

    def e_ai_26():
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.LSTM(4)])
        input_data = tf.random.uniform((1, 5))
        output = model(input_data)  # implicit: missing time dimension

    def e_ai_27():
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.pad_token = None
        tokens = tokenizer(["a", "b"], padding=True)  # implicit: no pad token set

    def e_ai_28():
        import torch
        x = torch.randn((5, 5), requires_grad=True)
        y = x.detach()
        y.backward(torch.ones_like(y))  # implicit: detached tensor cannot backward

    def e_ai_29():
        import numpy as np
        empty_array = np.array([])
        choice = np.random.choice(empty_array)  # implicit: cannot choose from empty array

    def e_ai_30():
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth="not_a_number")
        clf.fit([[1, 2], [3, 4]], [0, 1])  # implicit: invalid parameter type

    def e_ai_31():
        import tensorflow as tf
        import os
        model_dir = "missing_model_dir/"
        if not os.path.exists(model_dir):
            model_dir = model_dir.replace("missing", "not_found")
        model = tf.keras.models.load_model(model_dir)  # implicit: directory not found

    def e_ai_32():
        import torch
        import sys
        huge_tensor = torch.randn(int(1e15), int(1e15))
        torch.save(huge_tensor, "bigfile.pt")  # implicit: memory/serialization error

    def e_ai_33():
        import transformers
        model = transformers.AutoModel.from_pretrained(
            "bert-base-uncased",
            revision="unknown_tag_12345"
        )  # implicit: revision not found

    def e_ai_34():
        import pandas as pd
        df = pd.DataFrame({"x": [1, 2, 3]})
        df["bad"] = df["x"].apply(lambda v: v["bad"])  # implicit: int has no __getitem__

    def e_ai_35():
        pipeline_output = None
        if pipeline_output is None:
            processed = 1 / len(pipeline_output)  # implicit: NoneType has no len()

    # ======================================================
    #   ADD 65 NEW ADVANCED AI/ML ERROR SCENARIOS
    # ======================================================

    def e_ai_36():
        import torch
        import torchvision
        image = torch.randn(1, 4, 224, 224)  # 4 channels instead of 3
        model = torchvision.models.resnet18(pretrained=False)
        output = model(image)  # implicit: channel mismatch

    def e_ai_37():
        import torch.nn as nn
        import torch
        layer = nn.Linear(10, 5)
        x = torch.randn(5, 10)
        y = layer(x)
        z = nn.Conv2d(5, 16, 3)(y)  # implicit: Conv2d expects 4D tensor

    def e_ai_38():
        import tensorflow as tf
        encoder = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(20,))])
        decoder = tf.keras.Sequential([tf.keras.layers.Dense(20)])
        x = tf.random.uniform((1, 15))
        z = encoder(x)
        y = decoder(z)  # implicit: encoder output shape mismatch

    def e_ai_39():
        import torch
        from torch.optim import Adam
        model = torch.nn.Linear(5, 2)
        optimizer = Adam(model.parameters(), lr=-0.001)  # implicit: negative lr
        x = torch.randn(1, 5)
        loss = model(x).sum()

    def e_ai_40():
        import torch
        import torch.nn as nn
        embedding_layer = nn.Embedding(100, 50)
        indices = torch.LongTensor([0, 105, 50])  # 105 out of bounds
        embeddings = embedding_layer(indices)  # implicit: index out of range

    def e_ai_41():
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(50, 32, input_length=10),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        x = tf.random.uniform((1, 10), maxval=100, dtype=tf.int32)
        y = model(x)  # implicit: value 100 out of embedding range

    def e_ai_42():
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model = RandomForestRegressor(max_depth="invalid")
        model.fit(X, y)  # implicit: invalid parameter type

    def e_ai_43():
        import torch
        x = torch.randn(5, 5, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        z = (y ** 2).sum()
        z.backward()  # implicit: leaf variable used twice

    def e_ai_44():
        import tensorflow as tf
        x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
        with tf.GradientTape() as tape:
            y = tf.reduce_sum(x ** 2)
        grad = tape.gradient(y, x)
        grad = tape.gradient(y, x)  # implicit: tape already watched

    def e_ai_45():
        import sklearn.metrics as metrics
        y_true = [0, 1, 2]
        y_pred = [0, 2]
        score = metrics.accuracy_score(y_true, y_pred)  # implicit: length mismatch

    def e_ai_46():
        import torch
        from torch.nn.utils.rnn import pack_padded_sequence
        x = torch.randn(10, 20)
        lengths = torch.LongTensor([25, 15, 20])
        packed = pack_padded_sequence(x, lengths, enforce_sorted=False)  # implicit: length > sequence

    def e_ai_47():
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        x = tf.random.uniform((5, 5))
        y = tf.constant([0, 1, 2, 3, 4])
        model.fit(x, y, epochs=1)  # implicit: sparse loss on wrong dtype

    def e_ai_48():
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.hidden_size = -100  # implicit: negative dimension
        model = AutoModel.from_config(config)
        model(torch.randn(1, 10, config.hidden_size))

    def e_ai_49():
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.BatchNorm1d(10)  # wrong input size for batch norm
        )
        x = torch.randn(5, 10)
        y = model(x)  # implicit: BatchNorm1d dimension mismatch

    def e_ai_50():
        import numpy as np
        from sklearn.decomposition import PCA
        X = np.random.randn(5, 100)
        pca = PCA(n_components=150)  # more components than features
        X_transformed = pca.fit_transform(X)  # implicit: n_components > n_features

    def e_ai_51():
        import tensorflow as tf
        tf.debugging.assert_all_finite(tf.constant([1.0, float('inf')]), "Check failed")  # implicit: inf value

    def e_ai_52():
        import torch
        x = torch.randn(5, 5)
        indices = torch.LongTensor([10, 3, 1])
        y = x[indices]  # implicit: index out of bounds

    def e_ai_53():
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(['cat', 'dog', 'bird'])
        result = encoder.transform(['cat', 'elephant'])  # implicit: unseen label

    def e_ai_54():
        import torch.nn as nn
        import torch
        attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        q = torch.randn(10, 2, 64)
        k = torch.randn(10, 2, 32)  # wrong key dimension
        v = torch.randn(10, 2, 64)
        output = attention(q, k, v)  # implicit: key dimension mismatch

    def e_ai_55():
        import tensorflow as tf
        x = tf.random.uniform((10, 20))
        y = tf.nn.softmax(x, axis=5)  # implicit: axis out of range

    def e_ai_56():
        import torch
        from torch.nn import functional as F
        x = torch.randn(2, 3, 4, 5)
        y = F.adaptive_avg_pool2d(x, output_size=(10, 10))  # implicit: output size > input

    def e_ai_57():
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(X, y)
        knn.predict(np.random.randn(1, 5))  # implicit: n_neighbors > n_samples

    def e_ai_58():
        import torch
        import torch.optim as optim
        model = torch.nn.Linear(5, 2)
        optimizer = optim.SGD([])  # empty params
        x = torch.randn(1, 5)
        loss = model(x).sum()

    def e_ai_59():
        import tensorflow as tf
        @tf.function
        def process(x):
            return x + tf.Variable([1.0])  # implicit: variable in tf.function
        process(tf.constant([1.0, 2.0]))

    def e_ai_60():
        import torch
        from torch.nn.utils import weight_norm
        layer = torch.nn.Linear(5, 5)
        layer = weight_norm(layer, name="weight")
        layer = weight_norm(layer, name="weight")  # implicit: already applied

    def e_ai_61():
        import numpy as np
        X = np.random.randn(100, 50)
        cov = np.cov(X.T)
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues[0] = -1
        sqrt_eigs = np.sqrt(eigenvalues)  # implicit: negative eigenvalue

    def e_ai_62():
        import torch
        model = torch.nn.Linear(10, 5)
        state_dict = model.state_dict()
        state_dict['weight'] = torch.randn(3, 10)  # wrong shape
        model.load_state_dict(state_dict, strict=True)  # implicit: shape mismatch

    def e_ai_63():
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(10, 10))
        ])
        x = tf.random.uniform((1, 28, 28, 1))
        y = model(x)  # implicit: pool size > feature map

    def e_ai_64():
        import torch
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.sum(x)
        y.backward()
        z = torch.sum(x * 2)
        z.backward()  # implicit: gradient accumulation after backward

    def e_ai_65():
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        inputs = tokenizer(["a"] * 100000, return_tensors="pt")
        outputs = model(**inputs)  # implicit: token limit exceeded

    def e_ai_66():
        import torch
        import torch.distributions as dist
        normal = dist.Normal(0, -1)  # implicit: negative scale
        sample = normal.rsample()

    def e_ai_67():
        import numpy as np
        from scipy.optimize import minimize
        def f(x):
            return np.inf if x[0] < 0 else x[0] ** 2
        result = minimize(f, [1.0])  # implicit: inf in optimization

    def e_ai_68():
        import tensorflow as tf
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = x / 0  # implicit: integer division by zero

    def e_ai_69():
        import torch
        import torch.nn.functional as F
        input_data = torch.randn(2, 3, 32, 32)
        weight = torch.randn(64, 3, 1, 1)
        output = F.conv2d(input_data, weight, stride=0)  # implicit: invalid stride

    def e_ai_70():
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LinearRegression
        X = [[1, 2], [3, 4]]
        y = [1, 2]
        scores = cross_val_score(LinearRegression(), X, y, cv=10)  # implicit: cv > n_samples

    def e_ai_71():
        import torch
        x = torch.randn(5, 5, dtype=torch.float32)
        y = torch.randn(5, 5, dtype=torch.float64)
        z = torch.mm(x, y)  # implicit: dtype mismatch in mm

    def e_ai_72():
        import tensorflow as tf
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
        def model(x):
            return x + 1
        model(tf.random.uniform((2, 5)))  # implicit: shape mismatch

    def e_ai_73():
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = np.array([[1, 2], [1, 2]])
        scaler.fit(X)
        result = scaler.transform(X)  # implicit: zero variance

    def e_ai_74():
        import torch
        x = torch.randn(10, dtype=torch.float32)
        mask = torch.tensor([True, False, True])  # shape mismatch
        y = x[mask]  # implicit: mask size mismatch

    def e_ai_75():
        import tensorflow as tf
        x = tf.sparse.SparseTensor([[0, 0], [1, 1]], [1.0, 2.0], [2, 2])
        y = tf.dense_to_sparse(x)  # implicit: already sparse

    def e_ai_76():
        import torch
        from torch.nn import RNNCell
        rnn = RNNCell(input_size=10, hidden_size=20)
        x = torch.randn(5, 10)
        h = torch.randn(5, 15)  # wrong hidden size
        h_new = rnn(x, h)  # implicit: hidden size mismatch

    def e_ai_77():
        import pandas as pd
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y = [[1, 2], [2, 3]]
        mlb.fit(y)
        y_transformed = mlb.transform([[1, 5]])  # implicit: unseen class

    def e_ai_78():
        import torch
        x = torch.randn(3, 3)
        eigenvalues, eigenvectors = torch.linalg.eigh(x)
        wrong_reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        diff = torch.abs(wrong_reconstructed - x)
        if torch.any(diff > 1e10):
            _ = 1 / 0  # implicit: numerical instability

    def e_ai_79():
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(100, 32),
            tf.keras.layers.Dense(10)
        ])
        x = tf.constant([[1, 2], [3, 4]])
        model(x)
        model.summary()  # implicit: no input shape info in first call

    def e_ai_80():
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(10, 5)
        y = torch.randn(12)  # wrong size
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset)
        for batch in loader:
            pass  # implicit: size mismatch

    def e_ai_81():
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('scaler2', StandardScaler()),  # duplicate name
            ('model', LinearRegression())
        ])

    def e_ai_82():
        import tensorflow as tf
        x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
        with tf.GradientTape() as tape:
            tape.stop_recording()
            y = x * 2
        grad = tape.gradient(y, x)  # implicit: stopped recording

    def e_ai_83():
        import torch
        model = torch.nn.Linear(5, 3)
        x = torch.randn(2, 5)
        y = model(x)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()  # implicit: backward not called

    def e_ai_84():
        from transformers import pipeline
        nlp = pipeline("sentiment-analysis", model="nonexistent-model-xyz")  # implicit: model not found

    def e_ai_85():
        import torch
        x = torch.randn(5, 5)
        y = torch.topk(x, 10)  # implicit: k > tensor size

    def e_ai_86():
        import tensorflow as tf
        x = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
        y = tf.reduce_mean(x)  # implicit: mean on int type

    def e_ai_87():
        import numpy as np
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.cross(a, b)  # implicit: 1D cross product invalid

    def e_ai_88():
        import torch
        import torch.nn.functional as F
        x = torch.randn(2, 3, 4, 5)
        y = F.interpolate(x, size=(10, -5))  # implicit: negative size

    def e_ai_89():
        import tensorflow as tf
        x = tf.range(10)
        y = tf.repeat(x, repeats=0)  # implicit: repeat 0 times

    def e_ai_90():
        import sklearn.ensemble as ens
        clf = ens.VotingClassifier(
            estimators=[("a", None), ("b", None)]
        )
        clf.fit([[1, 2], [3, 4]], [0, 1])  # implicit: None estimators

    def e_ai_91():
        import torch
        x = torch.randn(10, 10)
        U, S, V = torch.svd(x)
        reconstructed = U @ torch.diag(S) @ V.T  # shape mismatch in SVD
        if reconstructed.shape != x.shape:
            _ = 1 / 0

    def e_ai_92():
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(16)
        ])
        x = tf.random.uniform((5, 10))  # missing time and feature dims
        y = model(x)  # implicit: wrong input shape

    def e_ai_93():
        import torch
        from torch.nn import TransformerEncoderLayer
        encoder = TransformerEncoderLayer(d_model=64, nhead=8)
        x = torch.randn(10, 2, 63)  # wrong d_model
        y = encoder(x)  # implicit: d_model mismatch

    def e_ai_94():
        import numpy as np
        from scipy.special import softmax
        x = np.array([1, 2, 3])
        y = softmax(x, axis=5)  # implicit: axis out of range

    def e_ai_95():
        import torch
        x = torch.randn(5, 5, requires_grad=True)
        y = x.sum()
        y.backward()
        z = y.detach()
        w = (z * x).sum()
        w.backward()  # implicit: x still has gradient from first backward

    def e_ai_96():
        import tensorflow as tf
        import numpy as np
        x = tf.constant(np.inf)
        y = tf.math.log(x)
        z = tf.cast(y, tf.int32)  # implicit: inf to int cast

    def e_ai_97():
        import torch
        model = torch.nn.Linear(10, 5)
        x = torch.randn(2, 10, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        loss.backward()  # implicit: called twice on same graph

    def e_ai_98():
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)  # implicit: tokenizer/model mismatch

    def e_ai_99():
        import torch
        import torch.nn as nn
        x = torch.randn(5, 5)
        layer = nn.DropConnect(p=1.5)  # implicit: invalid dropout probability
        y = layer(x)

    def e_ai_100():
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        x = tf.random.uniform((32, 5))
        y = tf.one_hot(tf.random.uniform((32,), maxval=10, dtype=tf.int32), depth=3)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(x, y, epochs=1)  # implicit: label index out of depth range


    # ======================================================
    #   FINAL ROUTE: RUN ALL 100 AI EXCEPTIONS
    # ======================================================

    @app.get("/ai/run-all")
    def ai_run_all():

        all_funcs = [
            e_ai_01, e_ai_02, e_ai_03, e_ai_04, e_ai_05,
            e_ai_06, e_ai_07, e_ai_08, e_ai_09, e_ai_10,
            e_ai_11, e_ai_12, e_ai_13, e_ai_14, e_ai_15,
            e_ai_16, e_ai_17, e_ai_18, e_ai_19, e_ai_20,
            e_ai_21, e_ai_22, e_ai_23, e_ai_24, e_ai_25,
            e_ai_26, e_ai_27, e_ai_28, e_ai_29, e_ai_30,
            e_ai_31, e_ai_32, e_ai_33, e_ai_34, e_ai_35,
            e_ai_36, e_ai_37, e_ai_38, e_ai_39, e_ai_40,
            e_ai_41, e_ai_42, e_ai_43, e_ai_44, e_ai_45,
            e_ai_46, e_ai_47, e_ai_48, e_ai_49, e_ai_50,
            e_ai_51, e_ai_52, e_ai_53, e_ai_54, e_ai_55,
            e_ai_56, e_ai_57, e_ai_58, e_ai_59, e_ai_60,
            e_ai_61, e_ai_62, e_ai_63, e_ai_64, e_ai_65,
            e_ai_66, e_ai_67, e_ai_68, e_ai_69, e_ai_70,
            e_ai_71, e_ai_72, e_ai_73, e_ai_74, e_ai_75,
            e_ai_76, e_ai_77, e_ai_78, e_ai_79, e_ai_80,
            e_ai_81, e_ai_82, e_ai_83, e_ai_84, e_ai_85,
            e_ai_86, e_ai_87, e_ai_88, e_ai_89, e_ai_90,
            e_ai_91, e_ai_92, e_ai_93, e_ai_94, e_ai_95,
            e_ai_96, e_ai_97, e_ai_98, e_ai_99, e_ai_100
        ]

        for fn in all_funcs:
            try: fn()
            except: pass

        return {"status": "done", "executed": len(all_funcs)}
