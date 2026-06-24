from serve import load_model
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient

model = load_model()

sample = pd.DataFrame(
    [
        {
            "amount": 100.0,
            "txn_count_24h": 2.0,
            "amount_sum_24h": 500.0,
            "merchant_category_Electronics": 1.0,
            "merchant_category_Fuel": 0.0,
            "merchant_category_Grocery": 0.0,
            "merchant_category_Healthcare": 0.0,
            "merchant_category_Restaurant": 0.0,
            "merchant_category_Shopping": 0.0,
            "merchant_category_Travel": 0.0,
        }
    ]
)

pred = model.predict(sample)

print(type(pred))
print(pred)
print(pred.shape)
print(pred.dtype)

print(type(model))
print(model.metadata.run_id)
print(model._model_impl)
print(type(model._model_impl))


run_id = "53ee2abe7baf4b168a513b909ceb76f2"

client = MlflowClient()

for f in client.list_artifacts(run_id):
    print(f.path)
