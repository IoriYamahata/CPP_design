#! /bin/bash



set -e



in_dir=$1
out_dir=$2



cd ${out_dir}

gmx_mpi grompp -f  ${PROCEDURE_DIR}/mdp/NPT.mdp \
               -c  NVT.gro                      \
               -r  NVT.gro                      \
               -t  NVT.cpt                      \
               -p  ${in_dir}/input.top          \
               -o  NPT.tpr                      \
               -po NPT.mdp                      || { echo "NPT grompp failed, skipping."; exit 0; }

gmx_mpi mdrun -s   NPT.tpr   \
              -o   NPT.trr   \
              -cpo NPT.cpt   \
              -c   NPT.gro   \
              -e   NPT.edr   \
              -g   NPT.log   || { echo "NPT mdrun failed, skipping."; exit 0; }

rm -f NVT.gro      \
      NVT.tpr      \
      NVT.cpt      \
      NPT.mdp      \
      NPT.trr      \
      NPT.edr      \
      NPT_prev.cpt \
      NPT.log
