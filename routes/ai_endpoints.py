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
        try:
            df = pd.DataFrame({"age": [25, 32, None], "city": ["Beirut", "Paris", "Berlin"]})
            df["age"] = df["age"].astype(str)  # force mixed dtype
            from sklearn.preprocessing import StandardScaler
            StandardScaler().fit_transform(df)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_02():
        import pandas as pd
        try:
            expected = ["feature1", "feature2", "feature3"]
            incoming = pd.DataFrame({"feature1": [1], "feature2": [2]})
            for col in expected:
                if col not in incoming.columns:
                    raise KeyError(f"Missing required feature column: '{col}'")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_03():
        import numpy as np
        try:
            X = np.array([["one", "two"], ["three", "four"]])
            from sklearn.linear_model import LinearRegression
            LinearRegression().fit(X, [1, 2])
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_04():
        try: raise ValueError("Model signature mismatch: expected 12 features, got 10")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_05():
        try: raise RuntimeError("Incompatible runtime: model requires torch==2.1.0 but found torch==1.9.0")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_06():
        try: raise ValueError("Feature distribution drift detected for 'transaction_amount' (p=0.001 < 0.05)")
        except:
            log(clean_traceback(), level="WARNING"); raise

    def e_ai_07():
        import torch
        from torch import nn
        try:
            model = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(32, 10),  # mismatch
            )
            x = torch.randn(1, 128)
            model(x)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_08():
        import tensorflow as tf
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, input_shape=(10,)),
                tf.keras.layers.Dense(4)
            ])
            model(tf.random.uniform((1, 8)))
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_09():
        import torch
        from torch import nn
        try:
            model = nn.Linear(10, 2)
            bogus = {"weight": torch.rand((3, 3)), "bias": torch.rand(4)}
            model.load_state_dict(bogus)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_10():
        import torch
        try:
            x = torch.tensor([1.0], requires_grad=False)
            (x * 2).backward()
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_11():
        import torch
        from torch import nn
        try:
            pred = torch.tensor([float("nan")])
            target = torch.tensor([1.0])
            loss = nn.MSELoss()(pred, target)
            if torch.isnan(loss):
                raise ValueError("Loss became NaN")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_12():
        import torch
        try: torch.load("non_existent_checkpoint.pt")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_13():
        from transformers import AutoTokenizer
        try: AutoTokenizer.from_pretrained("nonexistent-model-123")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_14():
        from transformers import BertModel
        try:
            model = BertModel.from_pretrained("bert-base-uncased")
            wrong = {"unexpected_key": 42}
            model.load_state_dict(wrong)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_15():
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            inputs = tokenizer("Hello", return_tensors="pt")
            model(**inputs, labels=torch.tensor([1]))  # wrong labels shape
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_16():
        try:
            from openai import OpenAI
            client = OpenAI()
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=-5
            )
        except:
            log(clean_traceback(), level="ERROR"); raise

    # ------------------------------------------------------
    #   ADD 20 NEW AI/ML REALISTIC ERRORS
    # ------------------------------------------------------

    def e_ai_17():
        import numpy as np
        try: np.linalg.inv(np.zeros((3,3)))
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_18():
        import pandas as pd
        try:
            df = pd.DataFrame({"x": ["a","b","c"]})
            from sklearn.preprocessing import MinMaxScaler
            MinMaxScaler().fit_transform(df)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_19():
        import tensorflow as tf
        try:
            x = tf.random.uniform((2,3))
            y = tf.random.uniform((4,5))
            _ = tf.matmul(x,y)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_20():
        try:
            import joblib
            joblib.load("missing_model.pkl")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_21():
        try:
            import torch
            torch.nn.Linear(5,5)(torch.randn(10,3))  # wrong input dim
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_22():
        import pandas as pd
        try:
            pd.read_csv("definitely_missing_file_abc123.csv")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_23():
        import numpy as np
        try:
            arr = np.arange(5)
            arr.reshape((3,3))
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_24():
        import torch
        try:
            torch.zeros((10,10), device="cuda:50")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_25():
        import pandas as pd
        try:
            df = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})
            from sklearn.impute import SimpleImputer
            SimpleImputer(strategy="mean").fit_transform(df) # mean on strings
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_26():
        import tensorflow as tf
        try:
            model = tf.keras.Sequential([tf.keras.layers.LSTM(4)])
            model(tf.random.uniform((1,5)))  # missing time dimension
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_27():
        import transformers
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
            tokenizer.pad_token = None
            tokenizer(["a","b"], padding=True)
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_28():
        import torch
        try:
            x = torch.randn((5,5), requires_grad=True)
            y = x.detach()
            y.backward(torch.ones_like(y))
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_29():
        import numpy as np
        try:
            np.random.choice([])
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_30():
        try:
            from sklearn.tree import DecisionTreeClassifier
            DecisionTreeClassifier(max_depth="not_a_number")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_31():
        import tensorflow as tf
        try:
            tf.keras.models.load_model("missing_model_dir/")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_32():
        import torch
        try:
            torch.save(torch.randn(10000000000000), "bigfile.pt")  # unrealistic object serialization
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_33():
        import transformers
        try:
            transformers.AutoModel.from_pretrained("bert-base-uncased", revision="unknown_tag")
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_34():
        import pandas as pd
        try:
            df = pd.DataFrame({"x":[1,2,3]})
            df["bad"] = df["x"].apply(lambda v: v["bad"])
        except:
            log(clean_traceback(), level="ERROR"); raise

    def e_ai_35():
        try: raise RuntimeError("Pipeline step produced empty output; aborting pipeline.")
        except:
            log(clean_traceback(), level="ERROR"); raise


    # ------------------------------------------------------
    #   FINAL ROUTE: RUN ALL AI EXCEPTIONS
    # ------------------------------------------------------

    @app.get("/ai/run-all")
    def ai_run_all():

        all_funcs = [
            e_ai_01, e_ai_02, e_ai_03, e_ai_04, e_ai_05,
            e_ai_06, e_ai_07, e_ai_08, e_ai_09, e_ai_10,
            e_ai_11, e_ai_12, e_ai_13, e_ai_14, e_ai_15,
            e_ai_16, e_ai_17, e_ai_18, e_ai_19, e_ai_20,
            e_ai_21, e_ai_22, e_ai_23, e_ai_24, e_ai_25,
            e_ai_26, e_ai_27, e_ai_28, e_ai_29, e_ai_30,
            e_ai_31, e_ai_32, e_ai_33, e_ai_34, e_ai_35
        ]

        for fn in all_funcs:
            try: fn()
            except: pass

        return {"status": "done", "executed": len(all_funcs)}
