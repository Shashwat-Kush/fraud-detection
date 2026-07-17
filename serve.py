import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import config
import mlflow
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Request
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import OneHotEncoder

from src.schemas import APITransaction

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"

SENDER_EMB_COLS = [f"sender_emb_{i}" for i in range(config.EMBEDDING_DIM)]
RECEIVER_EMB_COLS = [f"receiver_emb_{i}" for i in range(config.EMBEDDING_DIM)]


def load_model() -> PyFuncModel:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    return mlflow.pyfunc.load_model(model_uri=model_uri)


def predict(model: PyFuncModel, df: pd.DataFrame) -> np.ndarray:
    preds = model.predict(df)

    if not isinstance(preds, np.ndarray):
        preds = np.asarray(preds)

    # The tabular pyfunc wraps predict_proba and returns (n, 2);
    # the graph booster returns (n,). Normalize to P(fraud).
    if preds.ndim == 2 and preds.shape[1] == 2:
        preds = preds[:, 1]

    if preds.ndim != 1:
        raise ValueError(f"Expected 1D probability array, got shape {preds.shape}")

    return preds


def load_encoder(run_id: str) -> OneHotEncoder:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    client = MlflowClient()
    encoder_path = client.download_artifacts(
        run_id=run_id,
        path="encoder.pkl",
    )

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    if not isinstance(encoder, OneHotEncoder):
        raise TypeError(f"Expected OneHotEncoder, got {type(encoder)}")

    return encoder


def build_features(
    transaction: dict,
    encoder: OneHotEncoder,
    feature_cols: list[str],
    account_to_id: dict[str, int] | None = None,
    graph_embeddings: np.ndarray | None = None,
) -> pd.DataFrame:

    df = pd.DataFrame([transaction])

    encoded = encoder.transform(df[["merchant_category"]])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
        index=df.index,
    )

    parts = [df, encoded_df]

    if account_to_id is not None and graph_embeddings is not None:
        sender_idx = account_to_id.get(df.loc[0, "account_id"])
        receiver_idx = account_to_id.get(df.loc[0, "receiver_id"])

        parts.append(
            pd.DataFrame(
                {
                    "sender_in_graph": [float(sender_idx is not None)],
                    "receiver_in_graph": [float(receiver_idx is not None)],
                },
                index=df.index,
            )
        )

        if sender_idx is None:
            sender_embedding = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        else:
            sender_embedding = graph_embeddings[sender_idx]

        if receiver_idx is None:
            receiver_embedding = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        else:
            receiver_embedding = graph_embeddings[receiver_idx]

        parts.append(
            pd.DataFrame([sender_embedding], columns=SENDER_EMB_COLS, index=df.index)
        )
        parts.append(
            pd.DataFrame(
                [receiver_embedding], columns=RECEIVER_EMB_COLS, index=df.index
            )
        )

    df = pd.concat(parts, axis=1)
    df = df.drop(
        columns=[
            "receiver_id",
            "transaction_id",
            "account_id",
            "merchant_category",
            "is_fraud",
        ],
        errors="ignore",
    )

    missing_cols = set(feature_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required feature columns: {sorted(missing_cols)}")

    return df[feature_cols]


def write_log(log_path: Path, record: dict) -> None:
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        print(f"Failed to save record: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not 0 <= config.FRAUD_THRESHOLD <= 1:
        raise ValueError(f"Invalid FRAUD_THRESHOLD: {config.FRAUD_THRESHOLD}")

    app.state.model = load_model()

    run_id = app.state.model.metadata.run_id

    app.state.encoder = load_encoder(run_id)

    app.state.feature_cols = app.state.model.metadata.signature.inputs.input_names()

    # Graph mode is signature-driven: only load embeddings if the champion
    # was trained with them. The embedding matrix (~1.5 GB for PaySim) is
    # far too large for the MLflow artifact proxy, so it is read from the
    # processed-data directory; in production it would live in object
    # storage or a feature store.
    graph_mode = any(
        c.startswith(("sender_emb_", "receiver_emb_")) for c in app.state.feature_cols
    )

    if graph_mode:
        app.state.graph_embeddings = np.load(PROCESSED_DIR / "graph_embeddings.npy")

        with open(PROCESSED_DIR / "account_to_id.pkl", "rb") as f:
            app.state.account_to_id = pickle.load(f)
    else:
        app.state.graph_embeddings = None
        app.state.account_to_id = None

    yield


app = FastAPI(
    title="Fraud Detection Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/score")
async def score_transaction(
    transaction: APITransaction,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:

    features_df = build_features(
        transaction=transaction.model_dump(),
        encoder=request.app.state.encoder,
        feature_cols=request.app.state.feature_cols,
        account_to_id=request.app.state.account_to_id,
        graph_embeddings=request.app.state.graph_embeddings,
    )

    fraud_probability = float(predict(request.app.state.model, features_df)[0])

    is_fraud = int(fraud_probability >= config.FRAUD_THRESHOLD)

    log_record = features_df.iloc[0].to_dict()
    log_record["fraud_probability"] = fraud_probability
    log_record["is_fraud"] = is_fraud

    log_path = PROCESSED_DIR / "production_logs.jsonl"

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    background_tasks.add_task(
        write_log,
        log_path,
        log_record,
    )

    return {
        "transaction_id": transaction.transaction_id,
        "is_fraud": is_fraud,
        "fraud_probability": fraud_probability,
    }
