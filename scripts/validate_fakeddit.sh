#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec "${PYTHON:-python}" - "$@" <<'PY'
import sys
from pathlib import Path

from resource_paths import resolve_read_path
from run_gamed import build_parser
from scripts.run import launch

arguments = [*sys.argv[1:], '-train_dataset', 'fakeddit', '-test_dataset', 'fakeddit',
             '-network_arch', 'gamed', '-val', '1']
parser = build_parser()
parser.prog = 'validate_fakeddit.sh'
parser.description = 'Validate GAMED on Fakeddit; -checkpoint is required. Model, datasets and mode are fixed by this script.'
args = parser.parse_args(arguments)
if not args.checkpoint.strip():
    parser.error('-checkpoint must name a trained GAMED state-dict file')
checkpoint = Path(resolve_read_path(args.checkpoint))
if not checkpoint.is_file():
    parser.error('Checkpoint file not found: ' + str(checkpoint))
launch('run_gamed.py', arguments)
PY
