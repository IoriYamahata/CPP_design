#!/usr/bin/env bash

set -euo pipefail

in_dir=$1
out_dir=$2

mkdir -p "${out_dir}"

for pdb in "${in_dir}"/*.pdb; do
    if [[ -f "${pdb}" ]]; then
        base=$(basename "${pdb}" | sed -e 's/\.pdb$//')
        export SCHRODINGER_TMPDIR="${TMPDIR:-/tmp}/${base}"
        "${PROCEDURE_DIR}/scripts/make_GROMACS_input_main.sh" "${pdb}" "${out_dir}/${base}"
    fi
done
