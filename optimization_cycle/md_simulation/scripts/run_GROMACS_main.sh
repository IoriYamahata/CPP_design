#! /bin/bash



set -e



in_dir=$1
out_dir=$2



mkdir -p ${out_dir}

${PROCEDURE_DIR}/scripts/run_EM.sh  ${in_dir} ${out_dir}
${PROCEDURE_DIR}/scripts/run_NVT.sh ${in_dir} ${out_dir}
${PROCEDURE_DIR}/scripts/run_NPT.sh ${in_dir} ${out_dir}
${PROCEDURE_DIR}/scripts/run_MD.sh  ${in_dir} ${out_dir}
