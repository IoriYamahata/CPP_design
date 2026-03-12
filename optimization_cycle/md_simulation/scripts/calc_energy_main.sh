#!/usr/bin/env bash

set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

export PYTHONNOUSERSITE=1
unset PYTHONPATH

in_dir=$1
out_dir=$2

if [[ ! -f "${in_dir}/protein.tpr" || ! -f "${in_dir}/protein.trr" ]]; then
    echo "Missing protein.tpr/protein.trr in ${in_dir}. Skipping."
    exit 0
fi

mkdir -p "${out_dir}"
cd "${out_dir}"

python "${PROCEDURE_DIR}/scripts/make_ndx_command.py" --gro "${in_dir}/protein.gro"

${GMX_CMD:-gmx} make_ndx -f "${in_dir}/protein.gro" -o index.ndx < commands.in


${GMX_MMPBSA_CMD:-gmx_MMPBSA} -O            \
    -i  "${PROCEDURE_DIR}/mdp/mmpbsa.in"    \
    -cs "${in_dir}/protein.tpr"             \
    -ct "${in_dir}/protein.trr"             \
    -ci index.ndx                           \
    -cg 10 11                               \
    -o  energy.dat                          \
    -eo energy.csv                          \
    -nogui

rm -f _GMXMMPBSA*                           \
      COMPACT*                              \
      gmx_MMPBSA.log                        \
      COM_traj_0.xtc                        \
      COM.prmtop                            \
      LIG.prmtop                            \
      REC.prmtop                            \
      leap.log                              \
      commands.in                           \
      index.ndx
