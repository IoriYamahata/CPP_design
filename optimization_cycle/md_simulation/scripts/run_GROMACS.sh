#!/usr/bin/env bash

set -euo pipefail

in_dir=$1
out_dir=$2

omp_num_threads=$((MD_NUM_CPUS / MD_NUM_GPUS))
if [[ "${omp_num_threads}" -lt 1 ]]; then
    omp_num_threads=1
fi
export OMP_NUM_THREADS="${omp_num_threads}"

mkdir -p "${out_dir}"

idx=0
for base in $(ls "${in_dir}"); do
    GPU_ID=$((idx % MD_NUM_GPUS))
    export GMX_GPU_ID="${GPU_ID}"

    "${PROCEDURE_DIR}/scripts/run_GROMACS_main.sh" "${in_dir}/${base}" "${out_dir}/${base}" || {
        echo "MD failed for ${base}. Continuing with next."
        idx=$((idx + 1))
        continue
    }
    idx=$((idx + 1))
done
