import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "transactions"
SCORE_URL = os.getenv("SCORE_URL", "http://127.0.0.1:8000/score")

EXPERIMENT_NAME = "Fraud Detection"

TABULAR_MODEL_NAME = "fraud-detector"
GRAPH_MODEL_NAME = "fraud-detector-graph"

# Model served by the API. The tabular champion outperforms the graph
# model once leakage is removed, so it is the default; set MODEL_NAME to
# fraud-detector-graph to serve the graph variant.
MODEL_NAME = os.getenv("MODEL_NAME", TABULAR_MODEL_NAME)
MODEL_ALIAS = "champion"

# Sender accounts rarely recur in PaySim, so their embeddings mostly
# carry noise; set to 0 to train the graph model on receiver embeddings only.
INCLUDE_SENDER_EMBEDDINGS = os.getenv("INCLUDE_SENDER_EMBEDDINGS", "1") == "1"

# GraphSAGE embedding dimensionality (hidden_channels).
EMBEDDING_DIM = 64

# Fraction of each dataset kept for training in chronological splits.
TRAIN_FRAC = 0.8

# Decision threshold applied to P(fraud) at serving time.
# Can be changed without retraining or re-registering the model.
FRAUD_THRESHOLD = 0.95
