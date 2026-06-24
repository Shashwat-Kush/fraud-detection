import mlflow
import config
from mlflow.pyfunc import PyFuncModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from mlflow.tracking import MlflowClient
import pickle
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from src.schemas import APITransaction
from pathlib import Path
import json


def load_model() -> PyFuncModel:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    return mlflow.pyfunc.load_model(model_uri=model_uri)


def predict(model: PyFuncModel, df: pd.DataFrame) -> np.ndarray:
    preds = model.predict(df)
    if not isinstance(preds, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(preds)}")

    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError(f"Expected 2D array, got shape{preds.shape}")

    return preds[:, 1]


def score(model: PyFuncModel, df: pd.DataFrame, threshold: float) -> np.ndarray:
    preds = predict(model, df)
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    return (preds >= threshold).astype(int)


def load_encoder(run_id: str) -> OneHotEncoder:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    encoder_path = client.download_artifacts(run_id=run_id, path="encoder.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    if not isinstance(encoder, OneHotEncoder):
        raise TypeError(f"Expected OneHotEncoder, got {type(encoder)}")
    return encoder


def build_features(
    transaction: dict, encoder: OneHotEncoder, feature_cols: list[str]
) -> pd.DataFrame:
    df = pd.DataFrame([transaction])
    encoded = encoder.transform(df[["merchant_category"]])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(), index=df.index
    )
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(
        columns=["transaction_id", "account_id", "merchant_category", "is_fraud"],
        errors="ignore",
    )
    missing_cols = set(feature_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required feature columns: {sorted(missing_cols)}")
    return df[feature_cols]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    run_id = app.state.model.metadata.run_id
    app.state.encoder = load_encoder(run_id)
    app.state.feature_cols = app.state.model.metadata.signature.inputs.input_names()
    yield


app = FastAPI(title="Fraud Detection Service", version="1.0.0", lifespan=lifespan)


@app.post("/score")
async def score_transaction(transaction: APITransaction, request: Request) -> dict:
    features_df = build_features(
        transaction.model_dump(),
        request.app.state.encoder,
        request.app.state.feature_cols,
    )
    fraud_probability = float(predict(request.app.state.model, features_df)[0])
    log_record = features_df.iloc[0].to_dict()
    log_record["fraud_probability"] = fraud_probability
    project_root = Path(__file__).resolve().parent
    log_path = project_root / "data" / "processed" / "production_logs.jsonl"
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")
    except OSError as e:
        print(f"Failed to Save records, Error: {e}")
    is_fraud = int(
        score(request.app.state.model, features_df, config.FRAUD_THRESHOLD)[0]
    )
    log_record = features_df.iloc[0].to_dict()
    log_record["fraud_probability"] = fraud_probability
    return {
        "transaction_id": transaction.transaction_id,
        "is_fraud": is_fraud,
        "fraud_probability": fraud_probability,
    }
