#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

receptor_dir="${INPUT_DATA_DIR}/receptor"
final_dir="${DIFFDOCK_DIR}/result"
concat_pdb_script="${DIFFDOCK_DIR}/scripts/concat_pdb_files.py"

diffdock_output_dir="${2:-${DIFFDOCK_DIR}/result/diffdock_output}"

for ligand_file in "${diffdock_output_dir}"/*.pdb; do
    if [[ -f "${ligand_file}" ]]; then
        base_name=$(basename "${ligand_file}" .pdb)
        receptor_name=$(echo "${base_name}" | cut -d '_' -f 1)
        receptor_path="${receptor_dir}/${receptor_name}.pdb"
        output_file="${final_dir}/${base_name}.pdb"

        python "${concat_pdb_script}" "${receptor_path}" "${ligand_file}" "${output_file}"
    fi
done

echo "All PDB files concatenated successfully."
