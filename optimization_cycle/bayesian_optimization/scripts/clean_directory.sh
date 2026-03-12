#!/bin/bash

next_cycle_num=$((CYCLE_NUMBER + 1))
target_file="${INPUT_DATA_DIR}/input_csv/cycle_${next_cycle_num}_input_sequence.csv"

candidates_num=$(python -c "import json; file=open('${BAYESIAN_OPTIMIZATION_DIR}/config/optimize_config.json', 'r'); config=json.load(file); print(config['q']); file.close()")

if [ -f "$target_file" ] && [ $(wc -l < "$target_file") -eq $((candidates_num + 1)) ]; then
    echo "Next sequence file exists. Clean directory now..."

    md_data_dir=${MD_SIMULATION_DIR}/data/design
    md_source_dir=${MD_SIMULATION_DIR}/results
    diffdock_results_dir=${DIFFDOCK_DIR}/results 
    destination_dir=${CYCLE_RESULT_DIR}/MD_simulation_results

    if [ ! -d "$destination_dir" ]; then
        mkdir -p "$destination_dir"
    fi

    mv ${md_source_dir}/* ${destination_dir}/
    
    if [ -d "$diffdock_results_dir" ]; then
        rm -rf ${diffdock_results_dir}/*
    fi

    if [ -z "$(ls -A $md_source_dir)" ]; then
        rmdir "$md_source_dir"
        mkdir -p "${destination_dir}/0_pdb_files"
        find "$md_data_dir" -type f -exec mv {} "${destination_dir}/0_pdb_files/" \;
    else
        echo "Error: Some files or directories could not be moved." >&2
        exit 1
    fi

else
    echo "Target file does not exist or does not have exactly $candidates_num lines.">&2
    exit 1
fi
