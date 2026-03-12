#!/usr/bin/env bash

set -euo pipefail

# Project root can be overridden by PROJECT_DIR, otherwise use this file's directory.
project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

input_data_dir="${project_dir}/input_data"
diffdock_dir="${project_dir}/diffdock"
md_simulation_dir="${project_dir}/md_simulation"
prediction_dir="${project_dir}/prediction"
bayesian_optimization_dir="${project_dir}/bayesian_optimization"
results_dir="${project_dir}/results"

# Determine cycle number: use CYCLE_NUMBER if set, else latest cycle_* dir, else default to 1.
if [[ -n "${CYCLE_NUMBER:-}" ]]; then
    cycle_number="${CYCLE_NUMBER}"
else
    if [[ -d "${results_dir}" ]]; then
        latest_cycle_dir=$(find "${results_dir}" -maxdepth 1 -type d -name 'cycle_*' | sort -V | tail -n 1)
    else
        latest_cycle_dir=""
    fi

    if [[ -n "${latest_cycle_dir}" ]]; then
        cycle_number=$(basename "${latest_cycle_dir}" | cut -d'_' -f2)
    else
        cycle_number=1
    fi
fi

cycle_result_dir="${results_dir}/cycle_${cycle_number}"

export PROJECT_DIR="${project_dir}"
export INPUT_DATA_DIR="${input_data_dir}"
export DIFFDOCK_DIR="${diffdock_dir}"
export MD_SIMULATION_DIR="${md_simulation_dir}"
export PREDICTION_DIR="${prediction_dir}"
export BAYESIAN_OPTIMIZATION_DIR="${bayesian_optimization_dir}"
export RESULTS_DIR="${results_dir}"
export CYCLE_RESULT_DIR="${cycle_result_dir}"
export CYCLE_NUMBER="${cycle_number}"
export SUMMARY="${cycle_result_dir}/log/summary.txt"
