#!/usr/bin/env bash
# Generate the complete SUMO dataset (run on the server).
#   bash run_all.sh                 # 25 x 25 (config.yaml), 8 parallel workers
#   WORKERS=24 bash run_all.sh      # more workers on a bigger machine
#   ROWS=5 bash run_all.sh          # 5 x 5 test network
# Each step can also be run on its own; see README.md.
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-8}
SIZE=()
if [ -n "${ROWS:-}" ]; then SIZE=(--rows "$ROWS"); fi
PY=${PYTHON:-python}

echo "== 1/4 build network";            $PY build_network.py "${SIZE[@]}"
echo "== 2/4 calibrate density levels"; $PY calibrate_density.py "${SIZE[@]}" --workers "$W"
echo "== 3/4 vehicle trips";            $PY generate_demand.py "${SIZE[@]}"
echo "== 4/4 consumer-device data";     $PY generate_device_data.py "${SIZE[@]}" --workers "$W"
echo "done: data/ and results/"
