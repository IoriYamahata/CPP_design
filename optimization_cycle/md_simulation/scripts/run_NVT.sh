#! /bin/bash



set -e



in_dir=$1
out_dir=$2



cd ${out_dir}

gmx_mpi grompp -f  ${PROCEDURE_DIR}/mdp/NVT.mdp \
               -c  EM.gro                       \
               -r  EM.gro                       \
               -p  ${in_dir}/input.top          \
               -o  NVT.tpr                      \
               -po NVT.mdp                      || { echo "NVT grompp failed, skipping."; exit 0; }

gmx_mpi mdrun -s   NVT.tpr   \
              -o   NVT.trr   \
              -cpo NVT.cpt   \
              -c   NVT.gro   \
              -e   NVT.edr   \
              -g   NVT.log   || { echo "NVT mdrun failed, skipping."; exit 0; }

rm -f EM.gro       \
      EM.tpr       \
      NVT.mdp      \
      NVT.trr      \
      NVT.edr      \
      NVT_prev.cpt \
      NVT.log
