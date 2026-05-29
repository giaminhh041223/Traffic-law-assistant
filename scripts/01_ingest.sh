#!/usr/bin/env bash
# Convenience wrapper: ingest all configured PDFs.
set -e
cd "$(dirname "$0")/.."
python -m src.phase1_data_engineering.run_ingest --config configs/settings.yaml "$@"
