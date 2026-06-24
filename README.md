# Real-Time Fraud Detection & MLOps Pipeline

An end-to-end, event-driven Machine Learning architecture designed to detect financial fraud in real-time. This system processes streaming transactions, maintains stateful rolling-window features, serves predictions via an asynchronous API, and continuously monitors for statistical data drift.

Built using the **PaySim** financial dataset to simulate highly imbalanced, structurally accurate fraud typologies.

## 🏗 System Architecture

The pipeline is split into three distinct layers:

1. **Offline Batch Pipeline (Training & Data Engineering)**
   * **Ingestion:** Parses raw PaySim CSVs, aligns schemas, and chronologically splits data into historical and streaming datasets to prevent future leakage.
   * **Feature Engineering:** Calculates 24-hour rolling aggregations (`txn_count_24h`, `amount_sum_24h`) partitioned by `account_id` using strict left-closed windows to prevent target-time leakage.
   * **Model Training:** Trains calibrated **XGBoost** models. Implements threshold sweeping to handle severe class imbalance (<0.2% fraud).
   * **Model Registry:** Integrates with **MLflow** for artifact tracking, signature enforcement, and automated Champion/Challenger promotion based on AUC-PR metrics.

2. **Online Streaming Pipeline (Inference & Serving)**
   * **Event Broker:** Uses **Apache Kafka** to stream synthetic production data. Employs partition-key hashing (via `account_id`) to mathematically guarantee chronological ordering and prevent state corruption.
   * **Stream Processor:** A stateful Kafka Consumer that rebuilds dynamic account histories in memory and aligns online features perfectly with offline training distributions.
   * **Inference API:** A **FastAPI** microservice that loads the champion model, validates schema contracts via **Pydantic**, serves predictions, and safely logs network payloads to disk via asynchronous I/O.

3. **Observability Layer (Drift Monitoring)**
   * Analyzes production logs against baseline Parquet reference datasets using **Evidently AI**.
   * Evaluates early-warning leading indicators (Data Drift, Prediction Drift) using mathematically appropriate statistical tests (Jensen-Shannon for categoricals, Kolmogorov-Smirnov for numericals).

## 🛠 Technology Stack
* **Stream Processing:** Apache Kafka, Confluent Kafka Python
* **Model Serving:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas
* **MLOps & Tracking:** MLflow, Evidently AI
* **Infrastructure:** Docker, Docker Compose

## 🚀 Quickstart

### Prerequisites
* Python 3.11+
* Docker & Docker Compose
* Download the [PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the CSV in `data/raw/transactions.csv`.

### 1. Initialize Infrastructure
Start the Kafka broker, Zookeeper, and the FastAPI inference server.
```bash
docker-compose up -d
mlflow ui
uvicorn serve:app --reload --port 8000

### 2. Execute Offline Pipeline (Training)
Prepare the data, engineer features, and train the champion model.
```bash
python -m src.ingest_paysim
python -m src.features
python -m src.model
```

### 3. Start the Online Pipeline (Streaming)
Start the producer to simulate live traffic, followed by the stateful consumer.
```bash
python -m src.producer
# In a new terminal:
python -m src.consumer
```

### 4. Run Drift Diagnostics
Generate a statistical observability report comparing the live stream against the training baseline.
```bash
python -m src.monitor
```
*Open `data/processed/drift_report.html` in your browser to view the dashboard.*

## 🧠 Key Engineering Decisions
* **Strict Chronological Splits:** Prevented future leakage by abandoning random train/test splits in favor of strict time-series slicing.
* **Partitioned Stream Ordering:** Leveraged Kafka's partition keys to guarantee strict event-time ordering, eliminating race conditions in the consumer's rolling window state.
* **Schema Contract Enforcement:** Synchronized Pydantic schemas with MLflow's inferred model signatures to prevent silent feature dropping and `500 Internal Server Errors` at the API boundary
* **Leading vs. Lagging Indicators:** Architected the monitoring layer to track drift rather than waiting for delayed ground-truth labels (chargebacks), enabling real-time alerting.

## 🔮 Future Enhancements (Phase 5)
* Replace the tabular XGBoost architecture with a **Graph Neural Network (GNN)** using **PyTorch Geometric (GraphSAGE)** to evaluate the topological network of account interactions rather than isolated rolling windows.

***