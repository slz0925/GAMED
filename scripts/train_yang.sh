#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec "${PYTHON:-python}" "$PROJECT_ROOT/scripts/run.py" run_gamed.py "$@" \
    -train_dataset yang -test_dataset yang -network_arch gamed -val 0
