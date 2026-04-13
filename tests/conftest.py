import os
import sys

# Make `serving/` importable as a module from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))

# Point MLflow at a closed local port so connection attempts fail immediately
# (instead of blocking on DNS resolution for 'mlflow' or waiting for retries).
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
