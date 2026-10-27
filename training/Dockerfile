FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forecast_training/ forecast_training/

USER 10001

# Configuration comes from env (see k8s/retrain-cronjob.yaml) or CLI flags.
ENTRYPOINT ["python", "-m", "forecast_training"]
