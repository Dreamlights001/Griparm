#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/collect_demonstrations.py "$@"
