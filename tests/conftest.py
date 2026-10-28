import os
import sys

# Make `serving/` importable as a module from the repo root, and `training/`
# for forecast_training.features, which serving shares (the image copies it).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "training"))

# Point MLflow at a closed local port so connection attempts fail immediately
# (instead of blocking on DNS resolution for 'mlflow' or waiting for retries).
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
