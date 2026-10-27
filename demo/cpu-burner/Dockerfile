FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir prometheus-client==0.26.0
COPY app.py .
USER 10001
EXPOSE 8080
CMD ["python", "app.py"]
