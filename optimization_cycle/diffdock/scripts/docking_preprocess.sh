#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
input_csv_name="${2:-}"

if [[ -z "${input_csv_name}" ]]; then
    echo "Usage: docking_preprocess.sh <project_dir> <input_csv_name>"
    exit 1
fi

source "${project_dir}/directory_setting.sh"

input_csv_path="${INPUT_DATA_DIR}/input_csv/${input_csv_name}"
input_data_config="${INPUT_DATA_DIR}/config/input_data_config.json"

# DiffDock source directory (clone externally or set DIFFDOCK_ROOT)
diffdock_tool_dir="${DIFFDOCK_ROOT:-${DIFFDOCK_DIR}/DiffDock}"
if [[ ! -d "${diffdock_tool_dir}" ]]; then
    echo "Error: DiffDock directory not found at ${diffdock_tool_dir}."
    echo "       Clone DiffDock there or set DIFFDOCK_ROOT."
    exit 1
fi

mapfile -t receptor_list < <(python -c "import json; f=open('${input_data_config}'); data=json.load(f); f.close(); print('\\n'.join(data['target_receptor_list']))")
mapfile -t optimization_methods < <(python -c "import json; f=open('${input_data_config}'); data=json.load(f); f.close(); print('\\n'.join(data['optimization_method_list']))")

# error check
if [[ ! -f "${input_csv_path}" ]]; then
    echo "Error: ${input_csv_path} does not exist."
    exit 1
fi

if [[ ${#receptor_list[@]} -eq 0 ]]; then
    echo "Error: receptor_list is invalid. Check ${input_data_config}."
    exit 1
fi

if [[ ${#optimization_methods[@]} -eq 0 ]]; then
    echo "Error: optimization_method_list is empty. Check ${input_data_config}."
    exit 1
fi

for receptor_name in "${receptor_list[@]}"; do
    pdb_path="${INPUT_DATA_DIR}/receptor/${receptor_name}.pdb"
    if [[ ! -f "${pdb_path}" ]]; then
        echo "Error: PDB file for ${receptor_name}.pdb does not exist in ${INPUT_DATA_DIR}/receptor."
        exit 1
    fi
done

if [[ ${#receptor_list[@]} -ne ${#optimization_methods[@]} ]]; then
    echo "Error: The number of items in target_receptor_list and optimization_method_list does not match."
    exit 1
fi

receptor_dir="${INPUT_DATA_DIR}/receptor"
result_dir="${DIFFDOCK_DIR}/result"
diffdock_input_dir="${DIFFDOCK_DIR}/result/diffdock_input"
diffdock_output_dir="${DIFFDOCK_DIR}/result/diffdock_output"

mkdir -p "${result_dir}" "${diffdock_input_dir}" "${diffdock_output_dir}"

python "${DIFFDOCK_DIR}/scripts/make_diffdock_input.py" "${input_csv_path}" "${diffdock_input_dir}"

for protein_name in "${receptor_list[@]}"; do
    python "${DIFFDOCK_DIR}/scripts/create_input_csv.py" "${diffdock_input_dir}" "${protein_name}"
done

# Set DOCKING_NUMBER_PER_GPU from config for concat_input_csv.py
config_path="${DIFFDOCK_DIR}/config/diffdock_config.json"
export DOCKING_NUMBER_PER_GPU=$(python -c "import json; print(json.load(open('${config_path}'))['docking_number_per_GPU'])")

python "${DIFFDOCK_DIR}/scripts/concat_input_csv.py"

rm -f "${diffdock_tool_dir}/data"/*_tmp.csv
rm -f "${diffdock_tool_dir}/data/diffdock_input.csv"
