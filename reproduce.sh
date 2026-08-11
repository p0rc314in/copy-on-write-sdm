#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
gpu="${GPU:-0}"

case "$stage" in
  prepare)
    python3 scripts/prepare_wikitext.py --output-root runs/data
    ;;
  occupancy|serving)
    python3 scripts/run_experiments.py "$stage" --gpu "$gpu"
    ;;
  all)
    python3 scripts/prepare_wikitext.py --output-root runs/data
    python3 scripts/run_experiments.py all --gpu "$gpu"
    ;;
  *)
    echo "usage: ./reproduce.sh [prepare|occupancy|serving|all]" >&2
    exit 2
    ;;
esac
