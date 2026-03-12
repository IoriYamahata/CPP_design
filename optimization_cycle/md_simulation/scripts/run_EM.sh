#! /bin/bash



set -e



in_dir=$1
out_dir=$2



cd ${out_dir}

gmx_mpi grompp -f  ${PROCEDURE_DIR}/mdp/EM.mdp \
               -c  ${in_dir}/input.gro         \
               -r  ${in_dir}/input.gro         \
               -p  ${in_dir}/input.top         \
               -o  EM.tpr                      \
               -po EM.mdp || { echo "EM grompp failed, skipping."; exit 0; }

gmx_mpi mdrun -s EM.tpr   \
              -o EM.trr   \
              -c EM.gro   \
              -e EM.edr   \
              -g EM.log   || { echo "EM mdrun failed, skipping."; exit 0; }

rm -f EM.mdp    \
      EM.trr    \
      EM.edr    \
      EM.log
