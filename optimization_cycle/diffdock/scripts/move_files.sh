#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

temp_dir="${MD_SIMULATION_DIR}/data/unpaired_pdb"
design_dir="${MD_SIMULATION_DIR}/data/design"
input_data_config="${INPUT_DATA_DIR}/config/input_data_config.json"

mkdir -p "${temp_dir}" "${design_dir}"

mv "${DIFFDOCK_DIR}/result"/*.pdb "${temp_dir}" 2>/dev/null || true

mapfile -t receptor_list < <(python -c "import json; f=open('${input_data_config}'); data=json.load(f); f.close(); print('\\n'.join(data['target_receptor_list']))")

declare -A sequences

for file in "${temp_dir}"/*_*.pdb; do
    filename=$(basename "${file}")
    receptor="${filename%%_*}"
    sequence="${filename#*_}"
    sequence="${sequence%.pdb}"
    sequences[$sequence]+="${receptor} "
done

for sequence in "${!sequences[@]}"; do
    all_present=true
    for receptor in "${receptor_list[@]}"; do
        if [[ ! ${sequences[$sequence]} =~ ${receptor} ]]; then
            all_present=false
            break
        fi
    done

    if ${all_present}; then
        for receptor in "${receptor_list[@]}"; do
            if [[ -f "${temp_dir}/${receptor}_${sequence}.pdb" ]]; then
                mv "${temp_dir}/${receptor}_${sequence}.pdb" "${design_dir}/"
            fi
        done
    fi
done
