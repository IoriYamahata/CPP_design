#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"
source "${MD_SIMULATION_DIR}/scripts/CONFIG" "${MD_SIMULATION_DIR}"

out_dir="${PROCEDURE_DIR}/results"

${PROCEDURE_DIR}/scripts/calc_energy.sh ${out_dir}/2_MD       \
                                        ${out_dir}/3_energy
