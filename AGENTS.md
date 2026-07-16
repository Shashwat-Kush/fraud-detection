# Fraud Detection MLOps Pipeline - Agent Instructions

## Project Overview
End-to-end real-time fraud detection pipeline with 3 layers:
1. **Offline Batch** (training, feature engineering, MLflow model registry with Champion/Challenger)
2. **Online Streaming** (Kafka producer/consumer, FastAPI inference API with GraphSAGE embeddings)
3. **Observability** (Evidently AI drift monitoring against reference dataset)

**Key data**: PaySim synthetic financial transactions (<0.2% fraud rate)

## Key Commands

### Infrastructure
```bash
make up                           # Start Kafka (KRaft) + MLflow via docker compose
make serve                        # Start FastAPI inference server on :8000
```

### Offline Pipeline (Training)
```bash
# 1. Ingest raw PaySim CSV -> historical.parquet + streaming.parquet (80/20 chronological split)
python -m src.ingest_paysim

# 2. Build 24h rolling window features (left-closed window to prevent leakage)
python -m src.features

# 3. Train baseline XGBoost + register to MLflow (champion/challenger)
python -m src.model

# 4. Graph pipeline: GraphSAGE -> embeddings -> graph-augmented XGBoost
python -m src.train_graph
python -m src.extract_embeddings
python -m src.prepare_graph_data
python -m src.train_graph_xgb
# (or `make train` / `make graph`)
```

### Tests & Lint
```bash
make test    # pytest tests/
make lint    # ruff check src serve.py config.py tests
```

### Online Pipeline (Streaming)
```bash
# Terminal 1: Start producer (streams streaming.parquet to Kafka topic "transactions")
python -m src.producer

# Terminal 2: Start consumer (builds rolling features, calls /score API)
python -m src.consumer
```

### Monitoring
```bash
# Generate Evidently drift report (compares production_logs.jsonl vs reference.parquet)
python -m src.monitor
# Open data/processed/drift_report.html
```

## Key Configuration
- **config.py**: single source of truth — `MLFLOW_TRACKING_URI`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`, `SCORE_URL`, `MODEL_NAME`, `MODEL_ALIAS`, `EMBEDDING_DIM`, `TRAIN_FRAC`, `FRAUD_THRESHOLD` (env-var overridable where it matters)
- **docker-compose.yml**: Kafka (KRaft, no Zookeeper) on `localhost:9092`, MLflow server on `localhost:5000` (sqlite backend `mlflow.db`)
- **MLflow**: Models registered as `fraud-detector` (tabular XGB) and `fraud-detector-graph` (GraphSAGE+XGB). The API serves `fraud-detector-graph` by default (`config.MODEL_NAME`)
- **Champion alias**: `champion` (promoted automatically if AUC-PR improves by >0.01 AND recall maintained — both pipelines)

## Key Files & Architecture

| File | Purpose |
|------|---------|
| `src/ingest_paysim.py` | Raw CSV → chronological 80/20 split to parquet |
| `src/features.py` | 24h rolling window features (left-closed, per account_id) |
| `src/data_prep.py` | Shared split/encoding logic used by both training pipelines |
| `src/model.py` | Baseline XGBoost + MLflow logging + Champion/Challenger promotion |
| `src/graph_builder.py` | Account graph + node features + weak labels (train window only) |
| `src/train_graph.py` | GraphSAGE training (PyG NeighborLoader) → checkpoint |
| `src/extract_embeddings.py` | Full-graph inference → `graph_embeddings.npy` + account map |
| `src/prepare_graph_data.py` | Joins graph embeddings to tabular features |
| `src/train_graph_xgb.py` | GraphSAGE embeddings + XGBoost → registers + promotes `fraud-detector-graph` |
| `serve.py` | FastAPI `/score` endpoint, loads champion model + encoder + embeddings from MLflow artifacts |
| `src/producer.py` | Streams `streaming.parquet` to Kafka (partition key = account_id) |
| `src/consumer.py` | Consumes Kafka, maintains in-memory 24h rolling state, calls `/score` |
| `src/monitor.py` | Evidently AI drift report (JS divergence for categorical, KS for numerical) |
| `src/schemas.py` | Pydantic models for Kafka (`Transaction`) and API (`APITransaction`) |

## Critical Implementation Details

1. **Chronological splits only** - No random train/test split (prevents future leakage)
2. **Kafka partition key = `account_id`** - Guarantees per-account event ordering for correct rolling window state
3. **Left-closed rolling window** (`closed="left"`) - Excludes current transaction from features
4. **Schema contract** - Pydantic models match MLflow model signature; drift monitor validates column parity
5. **Champion/Challenger logic** (model.py, train_graph_xgb.py): Promote if `candidate_auc_pr > champion_auc_pr + 0.01` AND `candidate_recall >= champion_recall`
6. **Threshold tuning** - Default threshold 0.95 (configurable in `config.FRAUD_THRESHOLD`) chosen via sweep for precision on imbalanced data
7. **Graph embeddings** - 64-dim GraphSAGE embeddings for sender/receiver accounts, loaded at API startup from MLflow artifacts
8. **Leakage-safe graph** - graph + weak node labels built from the training window only (`load_graph_window`), never from the test period
8. **Production logging** - Async background task writes scored transactions to `data/processed/production_logs.jsonl`

## Common Issues / Gotchas
- **PaySim CSV required** at `data/raw/transactions.csv` (download from Kaggle)
- **MLflow + Kafka must be up** before training/streaming (`make up`)
- **FastAPI must be running** before consumer (`make serve`)
- **Graph embeddings** saved as `.npy` + pickle maps; loaded at API startup via MLflow artifact download
- **Drift monitor fails** if `production_logs.jsonl` empty or missing columns vs reference set
- **MPS device** used for GraphSAGE if available (Mac M-series); falls back to CPU

## Verification Checklist
After full pipeline run:
- MLflow UI shows `fraud-detector-graph` model with `champion` alias
- `docker compose ps` shows kafka/mlflow healthy
- `curl http://localhost:8000/docs` shows API schema
- `python -m src.consumer` logs predictions with `is_fraud` and `fraud_probability`
- `data/processed/drift_report.html` opens with drift metrics