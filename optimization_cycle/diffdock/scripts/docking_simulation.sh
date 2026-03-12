#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

# DiffDock source directory (clone externally or set DIFFDOCK_ROOT)
diffdock_tool_dir="${DIFFDOCK_ROOT:-${DIFFDOCK_DIR}/DiffDock}"
if [[ ! -d "${diffdock_tool_dir}" ]]; then
    echo "Error: DiffDock directory not found at ${diffdock_tool_dir}."
    echo "       Clone DiffDock there or set DIFFDOCK_ROOT."
    exit 1
fi

config_path="${DIFFDOCK_DIR}/config/diffdock_config.json"
INFERENCE_STEPS=$(python -c "import json; print(json.load(open('${config_path}'))['inference_steps'])")
SAMPLES_PER_COMPLEX=$(python -c "import json; print(json.load(open('${config_path}'))['samples_per_complex'])")
BATCH_SIZE=$(python -c "import json; print(json.load(open('${config_path}'))['batch_size'])")
ACTUAL_STEPS=$(python -c "import json; print(json.load(open('${config_path}'))['actual_steps'])")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

diffdock_output_dir="${DIFFDOCK_DIR}/result/diffdock_output"

cd "${diffdock_tool_dir}"

shopt -s nullglob
input_files=("${diffdock_tool_dir}/data"/*_diffdock_input.csv)
if [[ ${#input_files[@]} -eq 0 ]]; then
    echo "Error: No *_diffdock_input.csv files found in ${diffdock_tool_dir}/data"
    exit 1
fi

for input_csv in "${input_files[@]}"; do
    echo "*** Starting inference ${input_csv} ***"
    python -m inference \
        --protein_ligand_csv    "${input_csv}" \
        --out_dir               "${diffdock_output_dir}" \
        --inference_steps       "${INFERENCE_STEPS}" \
        --samples_per_complex   "${SAMPLES_PER_COMPLEX}" \
        --batch_size            "${BATCH_SIZE}" \
        --actual_steps          "${ACTUAL_STEPS}" \
        --no_final_step_noise

done
