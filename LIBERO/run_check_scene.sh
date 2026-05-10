#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/check_scene.py
