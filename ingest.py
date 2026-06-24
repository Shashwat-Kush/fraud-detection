from src.simulator import (
    transaction_generator,
    generate_accounts,
    persona_metadata,
    persona_distribution,
    population,
)
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent
raw_data_dir = project_root / "data" / "raw"
raw_data_dir.mkdir(parents=True, exist_ok=True)

Output_file = raw_data_dir / "historical_transactions.parquet"
num_transactions = 50000
accounts = generate_accounts(population, persona_distribution, persona_metadata)
generator = transaction_generator(accounts)

records = []

for _ in range(num_transactions):
    txn = next(generator)
    records.append(txn.model_dump())

df = pd.DataFrame(records)

df.to_parquet(
    Output_file,
    index=False,
)

print(f"Saved {len(df):,} transactions to {Output_file}")
