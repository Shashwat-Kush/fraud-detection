MLFLOW_TRACKING_URI = "http://localhost:5000"

MODEL_NAME = "fraud-detector"
MODEL_ALIAS = "champion"

# Decision threshold applied to P(fraud) at serving time.
# Can be changed without retraining or re-registering the model.
FRAUD_THRESHOLD = 0.95
