import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset


def load_production_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Production log file not found : {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty. Generate some predictions first.")

    try:
        df = pd.read_json(path, lines=True)
    except ValueError as e:
        raise ValueError(f"Failed to parse JSONL file: {path}") from e

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    reference_path = project_root / "data" / "processed" / "reference.parquet"

    production_path = project_root / "data" / "processed" / "production_logs.jsonl"

    output_path = project_root / "data" / "processed" / "drift_report.html"

    reference_df = pd.read_parquet(reference_path)

    production_df = load_production_logs(production_path).rename(
        columns={"fraud_probability": "prediction"}
    )

    required_cols = set(reference_df.columns) - {"target"}

    missing_cols = required_cols - set(production_df.columns)

    if missing_cols:
        raise ValueError(f"Production data missing columns: {sorted(missing_cols)}")

    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_df,
        current_data=production_df,
    )

    snapshot.save_html(str(output_path))

    print(f"Drift report saved to {output_path}")
