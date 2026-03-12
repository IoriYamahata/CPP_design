#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

data_dir="${MD_SIMULATION_DIR}/results/3_energy"
output_file="${CYCLE_RESULT_DIR}/training_dataset.csv"

python "${PREDICTION_DIR}/scripts/make_training_input.py" "${data_dir}" "${output_file}"
bash "${PREDICTION_DIR}/scripts/concat_previous_cycle_input.sh" "${project_dir}"
