FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictive_scaler/ predictive_scaler/

RUN useradd --system --uid 10001 scaler
USER 10001

# 6000: gRPC (KEDA + health), 9100: Prometheus metrics
EXPOSE 6000 9100

ENTRYPOINT ["python", "-m", "predictive_scaler"]
