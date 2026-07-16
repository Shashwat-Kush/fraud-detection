import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
import pickle
import tempfile
from typing import Any
import mlflow.pyfunc

import config
from src.data_prep import chronological_split, load_features, prepare_datasets


class FraudModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model: Any):
        self.model = model

    def load_context(self, context) -> None:
        # no need
        pass

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
    ):
        return self.model.predict_proba(model_input)


def log_encoder_artifact(encoder: OneHotEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder_path = Path(tmp_dir) / "encoder.pkl"

        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)

        mlflow.log_artifact(encoder_path)


def build_reference_dataset(
    model,
    x: pd.DataFrame,
    y: pd.Series,
    path: Path,
) -> None:
    reference_df = x.copy()

    reference_df["target"] = y

    reference_df["prediction"] = model.predict_proba(x)[:, 1]

    reference_df.to_parquet(path)


if __name__ == "__main__":
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "processed" / "historical_features.parquet"
    df = load_features(data_path)
    train_df, test_df = chronological_split(df)
    x_train, y_train, x_test, y_test, encoder = prepare_datasets(
        train_df,
        test_df,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    assert x_train.columns.equals(x_test.columns)

    model = XGBClassifier()

    model.fit(x_train, y_train)

    baseline_pred = model.predict(x_test)
    baseline_prob = model.predict_proba(x_test)[:, 1]

    baseline_precision = precision_score(
        y_test,
        baseline_pred,
    )

    baseline_recall = recall_score(
        y_test,
        baseline_pred,
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_pred,
    )

    baseline_auc_pr = average_precision_score(
        y_test,
        baseline_prob,
    )

    print("\nBASELINE")
    print(f"Precision: {baseline_precision:.4f}")
    print(f"Recall:    {baseline_recall:.4f}")
    print(f"F1:        {baseline_f1:.4f}")
    print(f"AUC-PR:    {baseline_auc_pr:.4f}")

    # # ==================================================
    # # 2. WEIGHTED MODEL
    # # ==================================================

    weighted_model = XGBClassifier(scale_pos_weight=scale_pos_weight)

    weighted_model.fit(x_train, y_train)

    weighted_pred = weighted_model.predict(x_test)
    weighted_prob = weighted_model.predict_proba(x_test)[:, 1]

    weighted_precision = precision_score(
        y_test,
        weighted_pred,
    )

    weighted_recall = recall_score(
        y_test,
        weighted_pred,
    )

    weighted_f1 = f1_score(
        y_test,
        weighted_pred,
    )

    weighted_auc_pr = average_precision_score(
        y_test,
        weighted_prob,
    )
    print("\nWEIGHTED")
    print(f"Precision: {weighted_precision:.4f}")
    print(f"Recall:    {weighted_recall:.4f}")
    print(f"F1:        {weighted_f1:.4f}")
    print(f"AUC-PR:    {weighted_auc_pr:.4f}")

    print("\nTHRESHOLD SWEEP")

    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
        preds = (weighted_prob >= threshold).astype(int)

        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    threshold = 0.95

    threshold_pred = (weighted_prob >= threshold).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_pred,
    )

    threshold_recall = recall_score(
        y_test,
        threshold_pred,
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_pred,
    )

    threshold_auc_pr = average_precision_score(
        y_test,
        weighted_prob,
    )

    print("\nWEIGHTED @ 0.95")
    print(f"Precision: {threshold_precision:.4f}")
    print(f"Recall:    {threshold_recall:.4f}")
    print(f"F1:        {threshold_f1:.4f}")
    print(f"AUC-PR:    {threshold_auc_pr:.4f}")
    print("\nCONFUSION MATRIX")
    print(
        confusion_matrix(
            y_test,
            threshold_pred,
        )
    )
    build_reference_dataset(
        weighted_model,
        x_test,
        y_test,
        project_root / "data" / "processed" / "reference.parquet",
    )

    # # ==================================================
    # # 3. BASELINE MODEL + LOWER THRESHOLD
    # # ==================================================

    threshold_run_id = None
    # ==================================================
    # MLFLOW TRACKING
    # ==================================================
    baseline_signature = infer_signature(x_train, model.predict_proba(x_train))

    weighted_signature = infer_signature(x_train, weighted_model.predict_proba(x_train))

    # --------------------------------------------------
    # RUN 1: BASELINE
    # --------------------------------------------------

    baseline_threshold = 0.5

    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params(
            {
                "model_type": type(model).__name__,
                "scale_pos_weight": model.get_params()["scale_pos_weight"],
                "threshold": baseline_threshold,
            }
        )

        mlflow.set_tags(
            {
                "experiment_type": "baseline",
                "class_weighting_enabled": "false",
            }
        )

        mlflow.log_metrics(
            {
                "precision": baseline_precision,
                "recall": baseline_recall,
                "f1": baseline_f1,
                "auc_pr": baseline_auc_pr,
            }
        )

        mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudModel(model),
            signature=baseline_signature,
        )

        log_encoder_artifact(encoder)

    # --------------------------------------------------
    # RUN 2: WEIGHTED MODEL
    # --------------------------------------------------

    weighted_threshold = 0.95

    with mlflow.start_run(run_name=f"weighted_spw_{scale_pos_weight:.2f}"):
        mlflow.log_params(
            {
                "model_type": type(weighted_model).__name__,
                "scale_pos_weight": weighted_model.get_params()["scale_pos_weight"],
                "threshold": weighted_threshold,
            }
        )

        mlflow.set_tags(
            {
                "experiment_type": "weighted",
                "class_weighting_enabled": "true",
            }
        )

        mlflow.log_metrics(
            {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": weighted_f1,
                "auc_pr": weighted_auc_pr,
            }
        )

        mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudModel(weighted_model),
            signature=weighted_signature,
        )
        log_encoder_artifact(encoder)

    # --------------------------------------------------
    # RUN 3: THRESHOLD-ADJUSTED
    # --------------------------------------------------

    with mlflow.start_run(run_name=f"weighted_threshold_{threshold}") as run:
        threshold_run_id = run.info.run_id
        mlflow.log_params(
            {
                "model_type": type(weighted_model).__name__,
                "scale_pos_weight": weighted_model.get_params()["scale_pos_weight"],
                "threshold": threshold,
            }
        )

        mlflow.set_tags(
            {
                "experiment_type": "weighted_threshold",
                "class_weighting_enabled": "true",
            }
        )

        mlflow.log_metrics(
            {
                "precision": threshold_precision,
                "recall": threshold_recall,
                "f1": threshold_f1,
                "auc_pr": threshold_auc_pr,
            }
        )
        # Weighted model evaluated at threshold 0.95
        mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudModel(weighted_model),
            signature=weighted_signature,
        )
        log_encoder_artifact(encoder)

    assert threshold_run_id is not None

    client = MlflowClient()

    # ----------------------------------
    # Register candidate
    # ----------------------------------

    registered_model = mlflow.register_model(
        model_uri=f"runs:/{threshold_run_id}/model",
        name=config.TABULAR_MODEL_NAME,
    )

    candidate_version = registered_model.version

    print(f"Registered candidate version {candidate_version}")

    # ----------------------------------
    # Champion-Challenger Evaluation
    # ----------------------------------

    try:
        champion = client.get_model_version_by_alias(
            name=config.TABULAR_MODEL_NAME,
            alias="champion",
        )

        champion_run = client.get_run(champion.run_id)
        candidate_run = client.get_run(threshold_run_id)

        champion_auc_pr = champion_run.data.metrics["auc_pr"]
        champion_recall = champion_run.data.metrics["recall"]

        candidate_auc_pr = candidate_run.data.metrics["auc_pr"]
        candidate_recall = candidate_run.data.metrics["recall"]

        print("\nMODEL COMPARISON")
        print(
            f"Champion v{champion.version} | "
            f"AUC-PR={champion_auc_pr:.4f} | "
            f"Recall={champion_recall:.4f}"
        )

        print(
            f"Candidate v{candidate_version} | "
            f"AUC-PR={candidate_auc_pr:.4f} | "
            f"Recall={candidate_recall:.4f}"
        )

        should_promote = (
            candidate_auc_pr > champion_auc_pr + 0.01
            and candidate_recall >= champion_recall
        )

        if should_promote:
            client.set_registered_model_alias(
                name=config.TABULAR_MODEL_NAME,
                alias="champion",
                version=candidate_version,
            )

            print(f"Candidate promoted to champion (v{candidate_version})")

        else:
            print(f"Candidate rejected. Champion remains v{champion.version}")

    except RestException:
        # No champion exists yet

        client.set_registered_model_alias(
            name=config.TABULAR_MODEL_NAME,
            alias="champion",
            version=candidate_version,
        )

        print(f"No existing champion found. Assigned champion -> v{candidate_version}")
