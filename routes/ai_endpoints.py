from fastapi import FastAPI
from utils.logger import log
import traceback
from routes.ai_pipeline import AIPipeline

def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


def register_ai_routes(app: FastAPI):

    @app.get("/ai/pipeline-preprocessing-mismatch")
    def ai_pipeline_preprocessing():
        import pandas as pd
        try:
            df = pd.DataFrame({"age": [25, 32, None], "city": ["Beirut", "Paris", "Berlin"]})
            df["age"] = df["age"].astype(str)  # mixed dtype
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit_transform(df)
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/feature-missing-column")
    def ai_missing_column():
        import pandas as pd
        try:
            expected = ["feature1", "feature2", "feature3"]
            incoming = pd.DataFrame({"feature1": [1], "feature2": [2]})
            for col in expected:
                if col not in incoming.columns:
                    raise KeyError(f"Missing required feature column: '{col}'")
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/feature-wrong-dtype")
    def ai_wrong_dtype():
        import numpy as np
        try:
            X = np.array([["one", "two"], ["three", "four"]])
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, [1, 2])
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/deploy-missing-signature")
    def ai_deploy_missing_signature():
        try:
            raise ValueError("Model signature mismatch: expected 12 features, got 10")
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/deploy-runtime-incompatible")
    def ai_deploy_runtime_incompatible():
        try:
            raise RuntimeError("Incompatible runtime: model requires torch==2.1.0 but found torch==1.9.0")
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/monitoring-drift-detected")
    def ai_monitoring_drift():
        try:
            raise ValueError("Feature distribution drift detected for 'transaction_amount' (p=0.001 < 0.05)")
        except Exception:
            log(clean_traceback(), level="WARNING")
            raise

    @app.get("/ai/nn-malformed")
    def ai_nn_malformed():
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
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/nn-invalid-input-size")
    def ai_nn_input_size():
        import tensorflow as tf
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, input_shape=(10,)),
                tf.keras.layers.Dense(4)
            ])
            x = tf.random.uniform((1, 8))
            model(x)
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/nn-weight-mismatch")
    def ai_nn_weight_mismatch():
        import torch
        from torch import nn
        try:
            model = nn.Linear(10, 2)
            fake_state = {"weight": torch.rand((3, 3)), "bias": torch.rand(4)}
            model.load_state_dict(fake_state)
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/nn-gradient-error")
    def ai_gradient_error():
        import torch
        try:
            x = torch.tensor([1.0], requires_grad=False)
            y = x * 2
            y.backward()
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/nn-loss-nan")
    def ai_loss_nan():
        import torch
        from torch import nn
        try:
            loss_fn = nn.MSELoss()
            pred = torch.tensor([float('nan')])
            target = torch.tensor([1.0])
            loss = loss_fn(pred, target)
            if torch.isnan(loss):
                raise ValueError("Loss value became NaN during training")
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/model-load-fail")
    def ai_model_load_fail():
        import torch
        try:
            torch.load("non_existent_checkpoint.pt")
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/tokenizer-error")
    def ai_tokenizer_error():
        from transformers import AutoTokenizer
        try:
            AutoTokenizer.from_pretrained("nonexistent-model-123")
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/huggingface-weight-mismatch")
    def ai_hf_weight_mismatch():
        from transformers import BertModel
        try:
            model = BertModel.from_pretrained("bert-base-uncased")
            wrong_state = {"unexpected_key": 42}
            model.load_state_dict(wrong_state)
        except Exception:
            log(clean_traceback(), level="ERROR")
            raise


    @app.get("/ai/transformer-mismatch")
    def ai_transformer():
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            outputs = model(**inputs, labels=torch.tensor([1]))  # Incorrect label shape
        except:
            log(clean_traceback(), level="ERROR")
            raise


    @app.get("/ai/gpt-error")
    def ai_gpt():
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Hello!"}
                ],
                max_tokens=-5  # Invalid parameter
            )
        except:
            log(clean_traceback(), level="ERROR")
            raise

    @app.get("/ai/run-pipeline")
    def run_pipeline(
        kaggle_link: str = "https://www.kaggle.com/datasets/codebreaker619/salary-data-with-age-and-experience"
    ):
        """
        Run the AI pipeline using the given Kaggle dataset link.
        Example:
        GET /ai/run-pipeline?kaggle_link=https://www.kaggle.com/datasets/codebreaker619/salary-data-with-age-and-experience
        """
        try:
            # Extract the Kaggle dataset reference
            parts = kaggle_link.split("/datasets/")[-1].split("/")
            dataset_ref = (
                parts[0] + "/" + parts[1]
                if len(parts) > 1
                else "codebreaker619/salary-data-with-age-and-experience"
            )

            pipeline = AIPipeline(source="C:/Users/sleim/Downloads/archive/Employee_Salary_Dataset.csv")
            pipeline.loader.dataset_ref = dataset_ref
            pipeline.loader.file_name = "Employee_Salary_Dataset.csv"
            pipeline.run_pipeline()

            return {"status": "success", "dataset": dataset_ref}

        except Exception as e:
            log(str(e), level="ERROR")
            return {"status": "error", "message": str(e)}