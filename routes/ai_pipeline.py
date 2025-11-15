import traceback
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils.logger import log

KaggleApi = None  # Lazy import placeholder


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


class DataLoader:
    def __init__(self, source: str, dataset_ref: str = None, file_name: str = None):
        self.source = source
        self.dataset_ref = dataset_ref
        self.file_name = file_name

    def download_from_kaggle(self):
        """Download Kaggle dataset if credentials exist"""
        global KaggleApi

        kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_config):
            log("⚠️ Kaggle credentials not found — using local CSV instead.", level="WARNING")
            return self.source

        try:
            if KaggleApi is None:
                from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            log(f"⬇️ Downloading Kaggle dataset: {self.dataset_ref}", level="INFO")
            api.dataset_download_files(self.dataset_ref, path="data/", unzip=True, quiet=True)
            csv_path = os.path.join("data", self.file_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"File {self.file_name} not found after download")
            return csv_path

        except Exception as e:
            log(f"⚠️ Kaggle download failed: {e} — continuing with local file.", level="WARNING")
            return self.source

    def load(self):
        log("📥 Loading dataset...", level="INFO")
        try:
            csv_path = self.download_from_kaggle() if (self.dataset_ref and self.file_name) else self.source
            data = pd.read_csv(csv_path)
            log(f"✅ Loaded {len(data)} rows from {csv_path}", level="INFO")
            return data
        except Exception as e:
            log(f"❌ Error loading data: {e}", level="ERROR")
            return pd.DataFrame()


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean(self, data: pd.DataFrame):
        log("🧹 Cleaning data...", level="INFO")
        if "salary" not in data.columns or "age" not in data.columns:
            raise KeyError("Missing required columns 'salary' or 'age'")
        data = data.dropna(subset=["salary", "age"])
        if "city" in data.columns:
            data = pd.get_dummies(data, columns=["city"], drop_first=True)
        return data

    def scale(self, data: pd.DataFrame):
        log("📊 Scaling features...", level="INFO")
        try:
            scaled = self.scaler.fit_transform(data)
            return scaled
        except Exception as e:
            raise RuntimeError(f"Scaling failed: {str(e)}")


class FeatureEngineer:
    def create_features(self, data: pd.DataFrame):
        log("⚙️ Creating new features...", level="INFO")
        if "salary" in data.columns and "age" in data.columns:
            data["income_per_age"] = data["salary"] / (data["age"] + 1)
        return data


class ModelTrainer:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        log("🏋️ Training model...", level="INFO")
        if len(X) < 3:
            raise ValueError("Insufficient training samples")
        self.model.fit(X, y)
        log("✅ Model trained successfully", level="INFO")


class ModelDeployer:
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def deploy(self):
        log("🚀 Deploying model...", level="INFO")
        if not self.trainer.model:
            raise RuntimeError("No model available for deployment")
        log("✅ Model deployed successfully!", level="INFO")
        return True


class AIPipeline:
    def __init__(self, source):
        self.loader = DataLoader(source)
        self.preprocessor = Preprocessor()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.deployer = ModelDeployer(self.trainer)

    def run_pipeline(self):
        log("🔄 Starting AI pipeline...", level="INFO")

        data = self.loader.load()
        if data.empty:
            log("⚠️ No data loaded — aborting pipeline.", level="WARNING")
            return

        try:
            data = self.preprocessor.clean(data)
            data = self.engineer.create_features(data)
            X = data.drop(columns=["salary"])
            y = data["salary"]
            X_scaled = self.preprocessor.scale(X)
            self.trainer.train(X_scaled, y)
            self.deployer.deploy()
            log("🎯 Pipeline completed successfully.", level="INFO")
        except Exception:
            log(clean_traceback(), level="ERROR")
