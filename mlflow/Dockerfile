# MLflow tracking server for production backends: the upstream image plus the
# PostgreSQL driver and the S3 client it does not ship. Same MLflow version as
# serving/ and training/ (the registry schema and the client must agree).
#
#   docker build -t mlflow-server mlflow/
FROM ghcr.io/mlflow/mlflow:v2.11.1

RUN pip install --no-cache-dir "psycopg2-binary==2.9.9" "boto3==1.34.106" \
    && useradd --uid 10001 --no-create-home mlflow

USER 10001
EXPOSE 5000
