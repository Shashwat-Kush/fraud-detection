.PHONY: up down train graph serve produce consume monitor test lint

up:            ## Start Kafka + MLflow
	docker compose up -d

down:
	docker compose down

train:         ## Offline pipeline: ingest -> features -> tabular XGBoost
	python -m src.ingest_paysim
	python -m src.features
	python -m src.model

graph:         ## Graph pipeline: GraphSAGE -> embeddings -> graph-augmented XGBoost
	python -m src.train_graph
	python -m src.extract_embeddings
	python -m src.prepare_graph_data
	python -m src.train_graph_xgb

serve:         ## Start the inference API
	uvicorn serve:app --port 8000

produce:       ## Stream transactions into Kafka
	python -m src.producer

consume:       ## Score the stream against the API
	python -m src.consumer

monitor:       ## Generate the Evidently drift report
	python -m src.monitor

test:
	pytest tests/

lint:
	ruff check .
