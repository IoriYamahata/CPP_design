#! /bin/bash



set -e




project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"
source "${MD_SIMULATION_DIR}/scripts/CONFIG" "${MD_SIMULATION_DIR}"


source ${procedure_dir}/scripts/CONFIG ${procedure_dir}



in_dir=${PROCEDURE_DIR}/data/design
out_dir=${PROCEDURE_DIR}/results/1_input

${PROCEDURE_DIR}/scripts/make_GROMACS_input.sh ${in_dir}  \
                                               ${out_dir}
