"""
Extended ML error scenarios (ml_err_101 - ml_err_180)
Adds 20 new errors for each ML category to enhance dataset diversity.
"""

def ml_err_101():
    """Schema evolution introduces unseen categorical level at inference."""
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    train = pd.DataFrame({"plan": ["basic", "pro"]})
    serve = pd.DataFrame({"plan": ["enterprise"]})
    enc = OneHotEncoder(handle_unknown="error").fit(train[["plan"]])
    enc.transform(serve[["plan"]]).toarray()


def ml_err_102():
    """Join between feature tables drops mandatory key column."""
    import pandas as pd
    a = pd.DataFrame({"user_id": [1, 2], "age": [20, 30]})
    b = pd.DataFrame({"uid": [1, 2], "spend": [100, 200]})
    pd.merge(a, b, on="user_id")


def ml_err_103():
    """StandardScaler divides by zero after constant column removal."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = np.ones((5, 3))
    StandardScaler(with_std=True).fit_transform(X)


def ml_err_104():
    """FeatureUnion returns arrays with inconsistent widths."""
    import numpy as np
    from sklearn.pipeline import FeatureUnion

    def f1(X):
        return np.ones((len(X), 2))

    def f2(X):
        return np.ones((len(X), 3))

    FeatureUnion([("f1", f1), ("f2", f2)]).fit_transform([[1], [2]])


def ml_err_105():
    """Rolling window imputer invoked with window larger than dataset."""
    import pandas as pd
    series = pd.Series([1.0])
    series.rolling(window=5).mean().ffill()


def ml_err_106():
    """Datetime parser rejects mix of timezone-aware and naive strings."""
    import pandas as pd
    pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02 01:00:00"])


def ml_err_107():
    """ColumnTransformer references column dropped by previous step."""
    from sklearn.compose import ColumnTransformer
    import pandas as pd
    df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"]})
    ColumnTransformer([("nums", "passthrough", ["num", "missing_col"])])\
        .fit_transform(df)


def ml_err_108():
    """QuantileTransformer fails on duplicated quantile probabilities."""
    import numpy as np
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=5, subsample=5, output_distribution="normal")
    qt.fit_transform(np.arange(5).reshape(-1, 1))


def ml_err_109():
    """PCA requested with more components than available samples."""
    import numpy as np
    from sklearn.decomposition import PCA
    PCA(n_components=5).fit_transform(np.random.rand(3, 4))


def ml_err_110():
    """MinMaxScaler receives inverted feature_range."""
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    MinMaxScaler(feature_range=(1, -1)).fit_transform(np.random.rand(5, 2))


def ml_err_111():
    """Graph adjacency matrix must be square."""
    import numpy as np
    np.linalg.inv(np.ones((3, 4)))


def ml_err_112():
    """Feature store snapshot violates monotonic timestamp expectation."""
    import pandas as pd
    df = pd.DataFrame({"ts": [2, 1], "value": [10, 20]})
    if not df["ts"].is_monotonic_increasing:
        raise RuntimeError("Feature snapshot timestamps not monotonic")


def ml_err_113():
    """Text vectorizer invoked before fitting vocabulary."""
    from sklearn.feature_extraction.text import CountVectorizer
    CountVectorizer().transform(["hello world"])


def ml_err_114():
    """JSON loader fails due to malformed document."""
    import json
    json.loads("{")


def ml_err_115():
    """Time-series resampler uses unsupported frequency alias."""
    import pandas as pd
    series = pd.Series(range(10), index=pd.date_range("2024-01-01", periods=10, freq="H"))
    series.resample("1X").mean()


def ml_err_116():
    """KBinsDiscretizer attempts to encode negative values with quantile strategy."""
    import numpy as np
    from sklearn.preprocessing import KBinsDiscretizer
    KBinsDiscretizer(strategy="quantile", n_bins=10).fit_transform(np.array([[-5.0], [-1.0], [0.5]]))


def ml_err_117():
    """Duplicate IDs detected after merge guard."""
    import pandas as pd
    left = pd.DataFrame({"id": [1, 1], "value": [10, 20]})
    right = pd.DataFrame({"id": [1], "flag": [True]})
    merged = left.merge(right, on="id")
    if merged["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected after merge")


def ml_err_118():
    """OrdinalEncoder receives category outside predefined order."""
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(categories=[["low", "medium", "high"]])
    enc.fit([["low"], ["medium"]])
    enc.transform([["critical"]])


def ml_err_119():
    """PowerTransformer invoked on data containing zeros using box-cox."""
    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    PowerTransformer(method="box-cox").fit_transform(np.array([[0.0], [1.0]]))


def ml_err_120():
    """Feature hashing dimension mismatch leads to index overflow."""
    from sklearn.feature_extraction import FeatureHasher
    vec = FeatureHasher(input_type="string", n_features=4).transform([{"feature": 1}]).todense()
    vec[0, 5]

# ============================================================
# Training & Optimization (ml_err_121 - ml_err_140)
# ============================================================


def ml_err_121():
    """GridSearchCV with refit=True but scorer returns NaN."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    import numpy as np

    def nan_scorer(estimator, X, y):
        return np.nan

    X, y = load_iris(return_X_y=True)
    GridSearchCV(SVC(), {"C": [0.1, 1.0]}, scoring=nan_scorer, refit=True).fit(X, y)


def ml_err_122():
    """RandomizedSearchCV receives distribution with invalid support."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    RandomizedSearchCV(LogisticRegression(), {"C": uniform(loc=-1, scale=0)}).fit([[1], [2]], [0, 1])


def ml_err_123():
    """GradientBoosting classifier fit with sample_weight mismatch."""
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    X = np.random.rand(10, 3)
    y = np.array([0, 1])
    GradientBoostingClassifier().fit(X, np.tile(y, 5), sample_weight=[1.0] * 3)


def ml_err_124():
    """RandomForest configured with invalid min_samples_leaf."""
    from sklearn.ensemble import RandomForestClassifier
    RandomForestClassifier(min_samples_leaf=0).fit([[1, 2], [3, 4]], [0, 1])


def ml_err_125():
    """TensorFlow compile invoked with unknown optimizer string."""
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(optimizer="unknown-optimizer", loss="mse")
    model.fit([[1, 2], [3, 4]], [0, 1])


def ml_err_126():
    """CrossEntropyLoss fed one-hot targets instead of class indices."""
    import torch
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
    torch.nn.CrossEntropyLoss()(logits, targets)


def ml_err_127():
    """TensorFlow estimator invoked with mismatched feature columns."""
    import tensorflow as tf
    feature_columns = [tf.feature_column.numeric_column("age")]
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    def input_fn():
        return {"missing": tf.constant([1.0])}, tf.constant([0])

    classifier.train(input_fn, steps=1)


def ml_err_128():
    """Pipeline partial_fit used on estimator lacking method."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    Pipeline([("scaler", StandardScaler()), ("clf", SGDClassifier())]).partial_fit([[1, 2]], [0])


def ml_err_129():
    """ParameterSampler receives invalid hyperparameter values."""
    from sklearn.model_selection import ParameterSampler
    list(ParameterSampler({"learning_rate": [0.1, -0.5]}, n_iter=2))


def ml_err_130():
    """Early stopping patience zero triggers validation failure."""
    patience = 0
    if patience <= 0:
        raise ValueError("Early stopping patience must be positive")


def ml_err_131():
    """TensorFlow custom training loop forgets to watch gradient tape."""
    import tensorflow as tf
    x = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        y = x * 2
    tape.gradient(y, x)


def ml_err_132():
    """Torch optimizer receives no parameters due to frozen model."""
    import torch
    model = torch.nn.Linear(4, 2)
    for param in model.parameters():
        param.requires_grad = False
    torch.optim.Adam(model.parameters()).step()


def ml_err_133():
    """Bagging classifier uses base estimator lacking random_state attribute."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVC
    BaggingClassifier(base_estimator=SVC(probability=False, random_state=None)).fit([[1]], [0])


def ml_err_134():
    """Keras fit called with mismatched sample weights length."""
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(optimizer="adam", loss="mse")
    model.fit([[1, 2], [3, 4]], [0, 1], sample_weight=[1.0])


def ml_err_135():
    """PyTorch distributed API queried before initialization."""
    import torch.distributed as dist
    dist.get_world_size()


def ml_err_136():
    """ONNX export fails when opset version unsupported."""
    import torch
    import torch.onnx as onnx
    onnx.export(torch.nn.Linear(2, 2), torch.randn(1, 2), "model.onnx", opset_version=1)


def ml_err_137():
    """TensorFlow checkpoint restoration fails due to missing variable."""
    import tensorflow as tf
    var = tf.Variable(1.0, name="weight")
    tf.train.Checkpoint(weight=var).restore("missing.ckpt").assert_existing_objects_matched()


def ml_err_138():
    """sklearn clone invoked on estimator with non-cloneable parameter."""
    from sklearn.base import clone
    clone(lambda x: x)


def ml_err_139():
    """Torch compile fails when backend is unsupported string."""
    import torch
    torch.compile(torch.nn.Linear(4, 2), backend="nonexistent")


def ml_err_140():
    """TensorFlow logical device configuration rejects unknown device type."""
    import tensorflow as tf
    tf.config.set_visible_devices([], device_type="QUANTUM")

# ============================================================
# Metrics & Evaluation (ml_err_141 - ml_err_160)
# ============================================================


def ml_err_141():
    """ROC AUC cannot be computed when only one class present."""
    from sklearn.metrics import roc_auc_score
    roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])


def ml_err_142():
    """Precision recall curve requires probabilities but receives labels."""
    from sklearn.metrics import precision_recall_curve
    precision_recall_curve([0, 1], [0, 1])


def ml_err_143():
    """Confusion matrix fails due to label mismatch."""
    from sklearn.metrics import confusion_matrix
    confusion_matrix([0, 1, 2], [0, 2])


def ml_err_144():
    """Cross_val_score invoked with cv greater than number of samples."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    y = (y == 0).astype(int)
    cross_val_score(LogisticRegression(), X[:4], y[:4], cv=5)


def ml_err_145():
    """Brier score receives probabilities outside [0,1]."""
    from sklearn.metrics import brier_score_loss
    brier_score_loss([0, 1], [1.2, -0.1])


def ml_err_146():
    """mean_absolute_percentage_error cannot handle zero targets."""
    from sklearn.metrics import mean_absolute_percentage_error
    mean_absolute_percentage_error([0, 0], [1, 2])


def ml_err_147():
    """log_loss invoked with probabilities that do not sum to one."""
    from sklearn.metrics import log_loss
    log_loss([0, 1], [[0.9, 0.2], [0.4, 0.1]])


def ml_err_148():
    """silhouette_score requires at least 2 clusters."""
    from sklearn.metrics import silhouette_score
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=5, centers=1, random_state=42)
    silhouette_score(X, [0, 0, 0, 0, 0])


def ml_err_149():
    """Matthews correlation fails with NaN predictions."""
    from sklearn.metrics import matthews_corrcoef
    matthews_corrcoef([0, 1], [0, float("nan")])


def ml_err_150():
    """balanced_accuracy_score invoked with empty arrays."""
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy_score([], [])


def ml_err_151():
    """Permutation importance fails when scoring function raises."""
    from sklearn.inspection import permutation_importance
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier().fit(X, y)

    def scorer(estimator, X, y):
        raise RuntimeError("Custom scorer failed")

    permutation_importance(model, X, y, scoring=scorer)


def ml_err_152():
    """Classification report fails when labels and target names length differ."""
    from sklearn.metrics import classification_report
    classification_report([0, 1], [0, 1], target_names=["negative"])


def ml_err_153():
    """Calibration curve fails because y_prob has wrong shape."""
    from sklearn.calibration import calibration_curve
    calibration_curve([0, 1], [[0.2, 0.8], [0.1, 0.9]])


def ml_err_154():
    """R-squared undefined when less than two samples provided."""
    from sklearn.metrics import r2_score
    r2_score([1], [1])


def ml_err_155():
    """F-beta score invoked with invalid beta parameter."""
    from sklearn.metrics import fbeta_score
    fbeta_score([0, 1], [0, 1], beta=-1)


def ml_err_156():
    """NDCG score cannot handle negative relevance values."""
    from sklearn.metrics import ndcg_score
    ndcg_score([[1, 0, 0]], [[-1, -2, -3]])


def ml_err_157():
    """Mean squared log error undefined for negative values."""
    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error([-1, 2], [0.5, 0.5])


def ml_err_158():
    """precision_recall_fscore_support fails when predictions miss class."""
    from sklearn.metrics import precision_recall_fscore_support
    precision_recall_fscore_support([0, 1], [0, 0], average="binary")


def ml_err_159():
    """Area-under-curve requires increasing x coordinates."""
    from sklearn.metrics import auc
    auc([1, 0, 2], [0, 1, 0])


def ml_err_160():
    """Custom score aggregator mismatches weight vector length."""
    scores = [0.8, 0.7]
    weights = [0.5]
    if len(scores) != len(weights):
        raise ValueError("Score aggregator weights mismatch")

# ============================================================
# Advanced & Production Scenarios (ml_err_161 - ml_err_180)
# ============================================================


def ml_err_161():
    """Model registry rejects artifact due to hash mismatch."""
    expected = "abc123"
    observed = "def456"
    if expected != observed:
        raise RuntimeError("Artifact hash mismatch detected")


def ml_err_162():
    """Feature drift job fails when reference window empty."""
    import pandas as pd
    reference = pd.Series([], dtype=float)
    if reference.empty:
        raise ValueError("Reference window empty for drift computation")


def ml_err_163():
    """Canary deploy rejects because shadow predictions missing."""
    shadow = None
    if shadow is None:
        raise RuntimeError("Shadow deployment produced no predictions")


def ml_err_164():
    """Streaming consumer lags behind offset retention period."""
    retention_ms = 60000
    lag_ms = 120000
    if lag_ms > retention_ms:
        raise RuntimeError("Consumer lag exceeds retention window")


def ml_err_165():
    """Ensemble router assigned zero weights but still invoked."""
    weights = []
    if not weights:
        raise ValueError("Ensemble router requires non-empty weights")


def ml_err_166():
    """Model governance check fails due to missing approval token."""
    approval_token = None
    if approval_token is None:
        raise PermissionError("Model lacks governance approval token")


def ml_err_167():
    """Batch scoring pipeline writes to read-only output path."""
    with open("/root/protected_predictions.csv", "w") as handle:
        handle.write("id,pred\n")


def ml_err_168():
    """Multi-tenant serving enforces tenant-specific encryption key."""
    tenant_keys = {"tenant-a": "k1"}
    tenant = "tenant-b"
    if tenant not in tenant_keys:
        raise KeyError("Tenant encryption key missing")


def ml_err_169():
    """A/B experiment aggregator receives inconsistent bucket sizes."""
    buckets = {"A": 100, "B": 80}
    if len(set(buckets.values())) != 1:
        raise RuntimeError("Buckets not balanced for experiment")


def ml_err_170():
    """Offline backfill attempts to rewrite already published partition."""
    partition_state = {"2024-01-01": "published"}
    target = "2024-01-01"
    if partition_state.get(target) == "published":
        raise PermissionError("Cannot overwrite published backfill partition")


def ml_err_171():
    """Policy guardrail prevents exporting model to unapproved region."""
    region = "us-west-2"
    allowed = {"eu-central-1"}
    if region not in allowed:
        raise RuntimeError("Model export blocked for unapproved region")


def ml_err_172():
    """Real-time feature service responds slower than SLO threshold."""
    latency_ms = 450
    if latency_ms > 200:
        raise TimeoutError("Feature service latency exceeded SLO")


def ml_err_173():
    """Metadata writer fails due to serialization of unsupported type."""
    import json
    json.dumps({"callback": lambda x: x})


def ml_err_174():
    """Drift retraining trigger fires but pipeline config missing."""
    pipeline_config = {}
    if "retrain" not in pipeline_config:
        raise KeyError("Retrain config missing for drift trigger")


def ml_err_175():
    """Data lineage tracker detects cycle in DAG definition."""
    dag = {"A": ["B"], "B": ["A"]}
    def dfs(node, stack):
        if node in stack:
            raise RuntimeError("Cycle detected in lineage graph")
        stack.add(node)
        for child in dag.get(node, []):
            dfs(child, stack)
        stack.remove(node)
    dfs("A", set())


def ml_err_176():
    """Model card publisher fails to upload artifact to storage."""
    raise IOError("Failed to upload model card to storage backend")


def ml_err_177():
    """Streaming feature join uses incompatible watermark semantics."""
    upstream_watermark = 100
    downstream_watermark = 50
    if downstream_watermark < upstream_watermark:
        raise RuntimeError("Watermark ordering violated for streaming join")


def ml_err_178():
    """Inference budget manager rejects request due to quota exhaustion."""
    quota = 1000
    used = 1200
    if used > quota:
        raise RuntimeError("Inference quota exhausted")


def ml_err_179():
    """Audit logger cannot redact sensitive field before persistence."""
    payload = {"ssn": "123-45-6789"}
    if "ssn" in payload:
        raise RuntimeError("Sensitive field present in audit payload")


def ml_err_180():
    """Blue/green deployment diff detects incompatible feature flags."""
    blue_flags = {"use_new_encoder": True}
    green_flags = {"use_new_encoder": False}
    if blue_flags != green_flags:
        raise RuntimeError("Feature flag mismatch between blue and green")


__all__ = [f"ml_err_{i:03d}" for i in range(101, 181)]
