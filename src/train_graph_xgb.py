import gc

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt

import config

NUM_BOOST_ROUND = 100
THRESHOLD = config.FRAUD_THRESHOLD
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"

TRAIN_FEATURES = DATA_DIR / "train_features.parquet"
TEST_FEATURES = DATA_DIR / "test_features.parquet"

TRAIN_LABELS = DATA_DIR / "y_train.npy"
TEST_LABELS = DATA_DIR / "y_test.npy"
ENCODER_PATH = DATA_DIR / "encoder.pkl"


class FraudModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        pass

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
    ):

        if isinstance(self.model, xgb.Booster):
            return self.model.predict(xgb.DMatrix(model_input))

        return self.model.predict_proba(model_input)


def evaluate(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float | np.ndarray]:
    pred = (probs >= threshold).astype(np.int8)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, pred),
        "pred": pred,
    }


def threshold_sweep(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:

    if thresholds is None:
        thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]

    results = []

    print("\nTHRESHOLD SWEEP")
    print("-" * 50)

    for threshold in thresholds:
        metrics = evaluate(y_true, probs, threshold)

        print(
            f"{threshold:.2f} | "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

        results.append(metrics)

    return results


if __name__ == "__main__":
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run(run_name="graph_xgboost"):
        mlflow.set_tags(
            {
                "model_type": "graph_xgboost",
                "graph_model": "GraphSAGE",
                "embedding_dim": "64",
                "merchant_feature": "True",
                "graph_embeddings": "True",
                "experiment": "baseline_plus_graph",
            }
        )
        print("=" * 60)
        print("GRAPH + XGBOOST TRAINING")
        print("=" * 60)

        print("\nLoading prepared datasets...")

        X_train = pd.read_parquet(TRAIN_FEATURES)
        X_test = pd.read_parquet(TEST_FEATURES)

        y_train = np.load(TRAIN_LABELS)
        y_test = np.load(TEST_LABELS)

        feature_columns = list(X_train.columns)
        assert feature_columns == list(X_test.columns)

        print(f"Train : {X_train.shape}")
        print(f"Test  : {X_test.shape}")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "max_bin": 256,
            "seed": 42,
            "nthread": 4,
            "verbosity": 1,
            "scale_pos_weight": scale_pos_weight,
        }

        mlflow.log_params(params)

        mlflow.log_params(
            {
                "sender_embedding_dim": 64,
                "receiver_embedding_dim": 64,
                "feature_count": len(feature_columns),
                "threshold": THRESHOLD,
                "num_boost_round": NUM_BOOST_ROUND,
            }
        )

        print("\nBuilding QuantileDMatrix...")

        dtrain = xgb.QuantileDMatrix(
            X_train,
            label=y_train,
        )
        signature_input = X_train.head(10).copy()
        del X_train
        gc.collect()

        print("Training model...")

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
        )

        del dtrain
        gc.collect()

        print("\nLoading test set...")

        dtest = xgb.DMatrix(X_test)

        del X_test
        gc.collect()

        probs = booster.predict(dtest)

        metrics = evaluate(
            y_test,
            probs,
            threshold=THRESHOLD,
        )

        print("\nGRAPH + XGBOOST RESULTS")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1        : {metrics['f1']:.4f}")
        print(f"AUC-PR    : {metrics['auc_pr']:.4f}")

        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("auc_pr", metrics["auc_pr"])
        mlflow.log_metric("threshold", THRESHOLD)

        sweep = threshold_sweep(
            y_test,
            probs,
        )
        sweep_df = pd.DataFrame(
            [
                {
                    "threshold": m["threshold"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "auc_pr": m["auc_pr"],
                }
                for m in sweep
            ]
        )

        sweep_path = DATA_DIR / "threshold_sweep.csv"

        sweep_df.to_csv(
            sweep_path,
            index=False,
        )

        mlflow.log_artifact(sweep_path)
        for result in sweep:
            thr = str(result["threshold"]).replace(".", "_")

            mlflow.log_metric(
                f"precision_at_{thr}",
                result["precision"],
            )

            mlflow.log_metric(
                f"recall_at_{thr}",
                result["recall"],
            )

            mlflow.log_metric(
                f"f1_at_{thr}",
                result["f1"],
            )

        print("\nConfusion Matrix @ 0.95")
        print(metrics["confusion_matrix"])

        cm = metrics["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(cm)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12,
                )

        plt.tight_layout()

        cm_path = DATA_DIR / "graph_confusion_matrix.png"

        fig.savefig(cm_path)

        plt.close(fig)
        gc.collect()
        mlflow.log_artifact(cm_path)

        pr_precision, pr_recall, _ = precision_recall_curve(
            y_test,
            probs,
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(pr_recall, pr_precision)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")

        plt.tight_layout()

        pr_path = DATA_DIR / "graph_pr_curve.png"

        fig.savefig(pr_path)

        plt.close(fig)
        gc.collect()

        mlflow.log_artifact(pr_path)

        importance_path = DATA_DIR / "feature_importance.csv"

        importance = booster.get_score(importance_type="total_gain")

        importance_df = pd.DataFrame(
            {
                "feature": list(importance.keys()),
                "gain": list(importance.values()),
            }
        ).sort_values(
            "gain",
            ascending=False,
        )
        importance_df.to_csv(
            importance_path,
            index=False,
        )
        mlflow.log_artifact(importance_path)

        signature_output = booster.predict(xgb.DMatrix(signature_input))

        signature = infer_signature(
            signature_input,
            signature_output,
        )

        wrapped_model = FraudModel(booster)
        # Only the encoder is small enough for the artifact store; serve.py
        # reads the account map and embeddings from the processed-data dir.
        mlflow.log_artifact(ENCODER_PATH)
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapped_model,
            signature=signature,
            registered_model_name=config.GRAPH_MODEL_NAME,
        )

    # ----------------------------------
    # Champion-Challenger Evaluation
    # ----------------------------------

    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(
            name=config.GRAPH_MODEL_NAME,
            alias=config.MODEL_ALIAS,
        )

        champion_metrics = client.get_run(champion.run_id).data.metrics

        should_promote = (
            metrics["auc_pr"] > champion_metrics["auc_pr"] + 0.01
            and metrics["recall"] >= champion_metrics["recall"]
        )

        print(
            f"Champion v{champion.version} | "
            f"AUC-PR={champion_metrics['auc_pr']:.4f} | "
            f"Recall={champion_metrics['recall']:.4f}"
        )

    except RestException:
        should_promote = True

    if should_promote:
        client.set_registered_model_alias(
            name=config.GRAPH_MODEL_NAME,
            alias=config.MODEL_ALIAS,
            version=model_info.registered_model_version,
        )

        print(f"Promoted v{model_info.registered_model_version} to champion")

    else:
        print("Candidate rejected. Champion unchanged.")
