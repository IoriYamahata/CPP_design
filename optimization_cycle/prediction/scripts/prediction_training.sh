#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

config_path="${PREDICTION_DIR}/config/prediction_config.json"
num_model=$(python -c "import json; print(json.load(open('${config_path}'))['num_model'])")

model_count=$(ls "${PREDICTION_DIR}/config/model_config_"* 2>/dev/null | wc -l | tr -d ' ')
if [[ "${model_count}" -eq 0 ]]; then
    echo "Error: no model_config_*.json found in ${PREDICTION_DIR}/config"
    exit 1
fi

# Use only the first model config for a lightweight test cycle.
model_id=1
num_trainings=${num_model}
if [[ "${num_trainings}" -lt 1 ]]; then
    num_trainings=1
fi

mkdir -p "${CYCLE_RESULT_DIR}/training_results"
mkdir -p "${CYCLE_RESULT_DIR}/prediction_results"

for training_id in $(seq 1 "${num_trainings}"); do
    python "${PREDICTION_DIR}/scripts/train.py" "${model_id}" "${training_id}"
done
