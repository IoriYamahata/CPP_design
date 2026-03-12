#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

result_dir="${DIFFDOCK_DIR}/result"
diffdock_output_dir="${result_dir}/diffdock_output"
receptor_dir="${INPUT_DATA_DIR}/receptor"

mkdir -p ${CYCLE_RESULT_DIR}

input_data_config="${INPUT_DATA_DIR}/config/input_data_config.json"
mapfile -t receptor_list < <(python -c "import json; f=open('${input_data_config}'); data=json.load(f); f.close(); print('\\n'.join(data['target_receptor_list']))")

amino_acids=("A" "C" "D" "E" "F" "H" "I" "L" "M" "N" "P" "Q" "S" "T" "V" "W" "Y"
             "RA" "RC" "RD" "RE" "RF" "RG" "RH" "RI" "RK" "RL" "RM" "RN" "RP" "RQ" "RR" "RS" "RT" "RV" "RW" "RY"
             "KA" "KC" "KD" "KE" "KF" "KG" "KH" "KI" "KK" "KL" "KM" "KN" "KP" "KQ" "KR" "KS" "KT" "KV" "KW" "KY"
             "GA" "GC" "GD" "GE" "GF" "GG" "GH" "GI" "GK" "GL" "GM" "GN" "GP" "GQ" "GR" "GS" "GT" "GV" "GW" "GY")

for receptor_name in "${receptor_list[@]}"; do
    for amino_acid in "${amino_acids[@]}"; do
        pattern_name="${receptor_name}_${amino_acid}"
        python "${DIFFDOCK_DIR}/scripts/convert_sdf_to_pdb.py" "${diffdock_output_dir}" "${pattern_name}"
    done
done


python  ${DIFFDOCK_DIR}/scripts/sort_pdb.py                  ${diffdock_output_dir}  
bash 	${DIFFDOCK_DIR}/scripts/concat_pdb_files.sh          ${project_dir} ${diffdock_output_dir}
bash    ${DIFFDOCK_DIR}/scripts/move_files.sh                ${project_dir}

mv      ${DIFFDOCK_DIR}/result                               ${CYCLE_RESULT_DIR}/
