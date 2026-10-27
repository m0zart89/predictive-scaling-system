#!/usr/bin/env bash
# Regenerate the Python gRPC stubs for the KEDA external scaler contract.
# Run from anywhere; requires grpcio-tools (see requirements-dev.txt).
# CI runs this and fails if the committed stubs differ from the output.
set -euo pipefail

cd "$(dirname "$0")"

python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  predictive_scaler/externalscaler.proto

echo "Stubs regenerated in scaler/predictive_scaler/"
