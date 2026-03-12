#!/bin/sh


set -e



project_dir=$1

source ${project_dir}/directory_setting.sh

output_file=${CYCLE_RESULT_DIR}/training_dataset.csv

last_cycle_num=$((CYCLE_NUMBER-1))

if [[ ${CYCLE_NUMBER} -gt 1 ]]; then
    previous_cycle_file="${RESULTS_DIR}/cycle_${last_cycle_num}/training_dataset.csv"
    if [ -f "$previous_cycle_file" ]; then
        tail -n +2 "$previous_cycle_file" >> $output_file

        echo -e "\n\033[92mPreprocessing Summary\033[0m"
        echo    " CSV files are combined"
        echo    "  Source     : $previous_cycle_file"
        echo    "  Destination: $output_file"
    else
        echo "Error: Previous cycle file $previous_cycle_file does not exist." >&2
        exit 1
    fi
fi