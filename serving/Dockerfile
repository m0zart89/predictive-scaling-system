# Build from the repository root (the transforms are shared with training/):
#   docker build -f serving/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Feature/target transforms shared with the retraining pipeline; numpy only.
COPY training/forecast_training/__init__.py training/forecast_training/features.py forecast_training/
COPY serving/app.py serving/quality.py ./

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
