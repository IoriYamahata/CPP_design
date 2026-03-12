#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

mkdir -p "${CYCLE_RESULT_DIR}/training_results"
mkdir -p "${CYCLE_RESULT_DIR}/prediction_results"

python "${PREDICTION_DIR}/scripts/concat_master_csv.py"
rm -f "${CYCLE_RESULT_DIR}/prediction_results"/*.csv.*
