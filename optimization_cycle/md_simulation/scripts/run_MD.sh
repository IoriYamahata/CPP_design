#! /bin/bash

set -euo pipefail

in_dir=$1
out_dir=$2

cd "${out_dir}"

# 初回だけ grompp して MD.tpr を作る
if [ ! -f MD.tpr ]; then
  gmx_mpi grompp -f  "${PROCEDURE_DIR}/mdp/MD.mdp" \
                 -c  NPT.gro                     \
                 -r  NPT.gro                     \
                 -t  NPT.cpt                     \
                 -p  "${in_dir}/input.top"       \
                 -o  MD.tpr                      \
                 -po MD.mdp                      || { echo "MD grompp failed, skipping."; exit 0; }
fi

gmx_mpi mdrun -s   MD.tpr \
              -o   MD.trr \
              -c   MD.gro \
              -e   MD.edr \
              -g   MD.log || { echo "MD mdrun failed, skipping."; exit 0; }

gmx_mpi trjconv -f MD.gro      \
                -s MD.tpr      \
                -o protein.gro <<- EOS
Protein
EOS
if [ $? -ne 0 ]; then echo "MD trjconv (gro) failed, skipping."; exit 0; fi

gmx_mpi trjconv -f   MD.trr      \
                -s   MD.tpr      \
                -o   protein.trr \
                -pbc nojump      <<- EOS
Protein
EOS
if [ $? -ne 0 ]; then echo "MD trjconv (trr) failed, skipping."; exit 0; fi

gmx_mpi make_ndx -f MD.gro      \
                 -o protein.ndx <<- EOS
q
EOS
if [ $? -ne 0 ]; then echo "MD make_ndx failed, skipping."; exit 0; fi

gmx_mpi convert-tpr -s MD.tpr      \
                    -n protein.ndx \
                    -o protein.tpr <<- EOS
Protein
EOS
if [ $? -ne 0 ]; then echo "MD convert-tpr failed, skipping."; exit 0; fi

rm -f NPT.gro     \
      NPT.tpr     \
      NPT.cpt     \
      MD.gro      \
      MD.tpr      \
      MD.mdp      \
      MD.trr      \
      MD.edr      \
      MD.log      \
      protein.ndx
