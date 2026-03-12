#!/usr/bin/env bash

set -euo pipefail

in_dir=$1
out_dir=$2

mapfile -t files < <(ls "${in_dir}")

for file in "${files[@]}"; do
    base=$(basename "${file}")
    "${PROCEDURE_DIR}/scripts/calc_energy_main.sh" "${in_dir}/${base}" "${out_dir}/${base}" || {
        echo "Error processing ${base}. Continuing with next file."
    }
done
