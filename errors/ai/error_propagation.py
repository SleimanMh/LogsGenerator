"""
AI Error Propagation Scenarios (e_ai_201 - e_ai_220)
Each function triggers failures that bounce through helper layers to
simulate multi-step call chains typical of production ML/AI stacks.
"""

import traceback

def _raise_with_trace(handler_name, exc):
    tb = "".join(traceback.format_exc())
    raise RuntimeError(f"{handler_name} failed -> {exc} | {tb}") from exc


def e_ai_201():
    """Feature store -> validator -> model mismatch cascade"""
    import pandas as pd

    def fetch_snapshot():
        df = pd.DataFrame({"age": [25], "salary": [70000]})
        missing_cols = {"tenure"} - set(df.columns)
        if missing_cols:
            raise KeyError(f"Snapshot missing columns: {missing_cols}")
        return df

    def validate(df):
        if df["salary"].iloc[0] < 0:
            raise ValueError("Salary must be non-negative")
        return df

    def infer(df):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit([[1, 2, 3]], [1])
        model.predict(df[["age", "salary", "tenure"]])

    try:
        infer(validate(fetch_snapshot()))
    except Exception as exc:
        _raise_with_trace("model_infer", exc)


def e_ai_202():
    """Pipeline metadata -> loader -> trainer path mismatch"""
    import os

    def resolve_artifact(meta):
        path = meta.get("path")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    def load_dataset(path):
        import pandas as pd
        return pd.read_csv(path)

    def train(df):
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(5,))])
        model.fit(df.values, df["target"], epochs=1)

    metadata = {"path": "non_existent_training_file.csv"}
    try:
        train(load_dataset(resolve_artifact(metadata)))
    except Exception as exc:
        _raise_with_trace("training_pipeline", exc)


def e_ai_203():
    """Tokenizer registry -> decoder -> metrics propagation"""
    from transformers import AutoTokenizer

    def get_tokenizer(name):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception as exc:
            _raise_with_trace("registry_lookup", exc)

    def decode(tokenizer):
        try:
            return tokenizer.decode([999999])
        except Exception as exc:
            _raise_with_trace("decoder", exc)

    def compute_metric(text):
        import numpy as np
        arr = np.array(text.split(), dtype=float)
        return arr.mean()

    decode(get_tokenizer("nonexistent-tokenizer-xyz"))
    compute_metric("1 2 3")


def e_ai_204():
    """GPU placement -> gradient hook -> scaler escalation"""
    import torch

    def allocate_on_gpu():
        return torch.zeros((1024, 1024), device="cuda:42")

    def register_hook(tensor):
        hooked = tensor.clone()
        hooked.register_hook(lambda grad: grad / 0)
        return hooked

    def scale_and_step(tensor):
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(tensor.sum()).backward()

    try:
        scale_and_step(register_hook(allocate_on_gpu()))
    except Exception as exc:
        _raise_with_trace("amp_scaler", exc)


def e_ai_205():
    """Autoencoder encoder->decoder shape contract"""
    import tensorflow as tf

    def build_encoder():
        return tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(4,))])

    def build_decoder():
        return tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(16,))])

    def run_autoencoder(x):
        latent = build_encoder()(x)
        return build_decoder()(latent)

    x = tf.random.uniform((1, 4))
    try:
        run_autoencoder(x)
    except Exception as exc:
        _raise_with_trace("autoencoder_pipeline", exc)


def e_ai_206():
    """Streaming chunk parser -> validator -> aggregator"""
    def parse_chunk(chunk):
        if "id" not in chunk:
            raise KeyError("chunk missing id")
        return chunk

    def validate(chunk):
        if chunk["sequence"] != sorted(chunk["sequence"]):
            raise ValueError("sequence ids out of order")
        return chunk

    def aggregate(chunk):
        import numpy as np
        return np.array(chunk["vectors"]).sum(axis=0)

    chunk = {"vectors": [[1, 2], [3, 4]], "sequence": [3, 2]}
    try:
        aggregate(validate(parse_chunk(chunk)))
    except Exception as exc:
        _raise_with_trace("stream_aggregator", exc)


def e_ai_207():
    """Model registry -> signature validator -> deployment"""
    def load_manifest(model_name):
        manifests = {}
        if model_name not in manifests:
            raise FileNotFoundError(f"manifest for {model_name} not found")
        return manifests[model_name]

    def validate_signature(manifest):
        required = {"inputs": 12}
        if manifest.get("inputs") != required["inputs"]:
            raise ValueError("model signature mismatch")
        return manifest

    def deploy(manifest):
        raise RuntimeError(f"deploy failed for {manifest}")

    try:
        deploy(validate_signature(load_manifest("fraud-detector-v2")))
    except Exception as exc:
        _raise_with_trace("deployment", exc)


def e_ai_208():
    """Data validation -> feature engineering -> trainer cascade"""
    import pandas as pd

    def validate(df):
        if df.isnull().any().any():
            raise ValueError("null values detected")
        return df

    def engineer(df):
        df["ratio"] = df["a"] / df["b"]
        return df

    def train(df):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(df[["a", "ratio"]], df["target"])

    payload = pd.DataFrame({"a": [1, 2], "b": [0, 0], "target": [0, 1]})
    try:
        train(engineer(validate(payload)))
    except Exception as exc:
        _raise_with_trace("feature_pipeline", exc)


def e_ai_209():
    """Prompt moderation -> formatter -> generator failure chain"""
    def moderate(prompt):
        blocklist = {"delete everything"}
        if any(term in prompt.lower() for term in blocklist):
            raise PermissionError("prompt violates policy")
        return prompt

    def format_prompt(prompt):
        return prompt.format(user="admin")

    def generate(prompt):
        from openai import OpenAI
        OpenAI().chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])

    prompt = "Delete everything for {user}"
    try:
        generate(format_prompt(moderate(prompt)))
    except Exception as exc:
        _raise_with_trace("llm_generation", exc)


def e_ai_210():
    """Config loader -> experiment planner -> distributed launch"""
    import json

    def load_config():
        config = "{"  # malformed JSON
        return json.loads(config)

    def plan_experiment(cfg):
        nodes = cfg["nodes"]
        if nodes % 2 != 0:
            raise RuntimeError("Need even worker count")
        return nodes

    def launch(nodes):
        raise RuntimeError(f"launcher failed for {nodes} nodes")

    try:
        launch(plan_experiment(load_config()))
    except Exception as exc:
        _raise_with_trace("distributed_launch", exc)


def e_ai_211():
    """Sensor ingestion -> kalman filter -> state estimator"""
    import numpy as np

    def ingest():
        return np.array([[1, 2], [3]])

    def kalman_step(obs):
        return np.linalg.inv(obs)

    def estimate(state):
        if not np.isfinite(state).all():
            raise ValueError("state vector invalid")

    try:
        estimate(kalman_step(ingest()))
    except Exception as exc:
        _raise_with_trace("state_estimator", exc)


def e_ai_212():
    """Embedding cache -> reranker -> decision service"""
    import numpy as np

    def fetch_embedding(doc_id):
        cache = {"doc-1": np.ones(3)}
        if doc_id not in cache:
            raise KeyError(doc_id)
        return cache[doc_id]

    def rerank(vec):
        import numpy as np
        weights = np.ones(vec.shape[0] + 1)
        return vec @ weights

    def decide(score):
        if np.isnan(score).any():
            raise ValueError("score contains NaN")

    try:
        decide(rerank(fetch_embedding("doc-1")))
    except Exception as exc:
        _raise_with_trace("decision_service", exc)


def e_ai_213():
    """Audio pipeline -> spectrogram -> classifier propagation"""
    import numpy as np

    def load_wave(path):
        if path != "valid.wav":
            raise FileNotFoundError(path)
        return np.zeros(10)

    def spectrogram(wave):
        import numpy as np
        window = np.hanning(2048)
        if wave.shape[0] < window.shape[0]:
            raise ValueError("window larger than audio buffer")
        return np.lib.stride_tricks.sliding_window_view(wave, window.shape[0]) @ window

    def classify(spec):
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit([[1, 2]], [0])
        clf.predict(spec.real)

    try:
        classify(spectrogram(load_wave("missing.wav")))
    except Exception as exc:
        _raise_with_trace("audio_classifier", exc)


def e_ai_214():
    """Vision ingest -> preprocessor -> detector chain"""
    import numpy as np

    def load_frame(path):
        import numpy as np
        if "missing" in path:
            return np.zeros((0, 0, 3))
        return np.random.rand(64, 64, 3)

    def preprocess(frame):
        if frame.ndim != 3:
            raise ValueError("Expected color image")
        return frame[:, :, ::-1]

    def detect(frame):
        if frame.size == 0:
            raise RuntimeError("Empty frame")
        if frame.shape[2] != 3:
            raise ValueError("Detector expects 3 channels")

    try:
        detect(preprocess(load_frame("missing_frame.jpg")))
    except Exception as exc:
        _raise_with_trace("vision_detector", exc)


def e_ai_215():
    """Temporal aligner -> feature stitcher -> predictor"""
    import pandas as pd

    def align(events):
        events = events.sort_values("ts")
        if events["ts"].duplicated().any():
            raise RuntimeError("duplicate timestamps")
        return events

    def stitch(events):
        pivot = events.pivot(index="ts", columns="sensor", values="value")
        return pivot.fillna(pivot.mean())

    def predict(df):
        from sklearn.cluster import KMeans
        KMeans(n_clusters=5).fit(df)

    events = pd.DataFrame({"ts": [1, 1], "sensor": ["a", "b"], "value": [5, 6]})
    try:
        predict(stitch(align(events)))
    except Exception as exc:
        _raise_with_trace("temporal_predictor", exc)


def e_ai_216():
    """Realtime monitor -> alert formatter -> pager duty escalator"""
    def monitor(metrics):
        if metrics.get("latency_ms", 0) > 200:
            raise RuntimeError("Latency SLO violated")
        return metrics

    def format_alert(metrics):
        return f"Latency={metrics['latency_ms']}ms"

    def send(alert):
        raise ConnectionError(f"Failed to send alert: {alert}")

    try:
        send(format_alert(monitor({"latency_ms": 250})))
    except Exception as exc:
        _raise_with_trace("alert_pipeline", exc)


def e_ai_217():
    """Model explainability -> attribution -> audit logger"""
    import numpy as np

    def compute_explanations(inputs):
        weights = np.array([0.1, 0.2])
        if inputs.shape[1] != weights.shape[0]:
            raise ValueError("Input dimension mismatch for explanations")
        return inputs @ weights

    def validate_attributions(attr):
        if np.isnan(attr).any():
            raise RuntimeError("Attribution contains NaN")
        return attr

    def log(attr):
        raise IOError("Audit log storage unavailable")

    data = np.random.rand(1, 3)
    try:
        log(validate_attributions(compute_explanations(data)))
    except Exception as exc:
        _raise_with_trace("attrib_audit", exc)


def e_ai_218():
    """Batch scorer -> aggregator -> notifier chain"""
    import numpy as np

    def score(batch):
        if batch.shape[0] == 0:
            raise ValueError("empty batch")
        return np.mean(batch, axis=1)

    def aggregate(scores):
        if (scores < 0).any():
            raise RuntimeError("negative scores detected")
        return scores.mean()

    def notify(summary):
        raise RuntimeError(f"Notification failed for summary {summary}")

    batch = np.zeros((0, 5))
    try:
        notify(aggregate(score(batch)))
    except Exception as exc:
        _raise_with_trace("batch_notifier", exc)


def e_ai_219():
    """Vector DB -> ANN search -> ranking chain"""
    import numpy as np

    def fetch_vectors():
        raise TimeoutError("vector DB timed out")

    def ann_search(vectors):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors()
        nn.fit(vectors)
        return nn.kneighbors([[0.1, 0.2]])

    def rank(results):
        import pandas as pd
        df = pd.DataFrame(results[0])
        return df.sort_values(0)

    try:
        rank(ann_search(fetch_vectors()))
    except Exception as exc:
        _raise_with_trace("ranking_service", exc)


def e_ai_220():
    """Monitoring hook -> drift detector -> retraining trigger"""
    import numpy as np

    def monitor():
        return {"population": np.array([0.1, 0.2]), "baseline": np.array([0.1, 0.2, 0.3])}

    def detect(payload):
        if payload["population"].shape != payload["baseline"].shape:
            raise ValueError("Distribution vectors mismatched")
        return payload

    def retrain(payload):
        raise RuntimeError(f"Retraining job failed for payload {payload}")

    try:
        retrain(detect(monitor()))
    except Exception as exc:
        _raise_with_trace("drift_retraining", exc)


__all__ = [f"e_ai_{i:03d}" for i in range(201, 221)]
