#!/usr/bin/env bash
# Convenience wrapper: build BM25 + ChromaDB indexes from Phase 1 output.
set -e
cd "$(dirname "$0")/.."
python -m src.phase2_hybrid_search.build_indexes \
    --settings configs/settings.yaml \
    --retrieval configs/retrieval.yaml \
    "$@"
