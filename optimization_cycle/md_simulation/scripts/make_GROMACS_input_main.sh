#! /bin/bash







PDB=$1
out_dir=$2




prepwizard() {
	input=$1
	output=$2

	if [[ -z "${SCHRODINGER:-}" || ! -x "${SCHRODINGER}/utilities/prepwizard" ]]; then
		echo "Error: SCHRODINGER is not set or prepwizard is missing." >&2
		echo "Set SCHRODINGER to your installation root (e.g., /path/to/schrodinger)." >&2
		exit 1
	fi

	job_ID=$(
		${SCHRODINGER}/utilities/prepwizard -rehtreat                \
		                                    -disulfides              \
		                                    -captermini              \
		                                    -cap_termini_min_atoms 0 \
		                                    -noepik                  \
		                                    -samplewater             \
		                                    -propka_pH 7.0           \
		                                    -noimpref                \
		                                    ${input}                 \
		                                    ${output}                \
		  |	sed -e 's/^JobId: //'
	)

	while [ -e .${job_ID} ]
	do
		sleep 5
	done
}


modify_GROMACS_file() {
	gro=$1
	top=$2

	sed -i -e "s/  IP  /  Na+ /g" ${top}
	sed -i -e "s/  IM  /  Cl- /g" ${top}
	sed -i -e "s/NA+/Na+/g"       ${top}
	sed -i -e "s/NA+/Na+/g"       ${gro}
	sed -i -e "s/CL-/Cl-/g"       ${top}
	sed -i -e "s/CL-/Cl-/g"       ${gro}
}



mkdir -p ${out_dir}
cd       ${out_dir}

${PROCEDURE_DIR}/scripts/serialize_resID.py --pdb ${PDB}     \
                                            >     serial.pdb

prepwizard serial.pdb prep.pdb
prepwizard serial.pdb prep.mae

${PROCEDURE_DIR}/scripts/amide_capping.py   --pdb prep.pdb   \
                                            >     capped.pdb

${PROCEDURE_DIR}/scripts/convert_PDB.py --pdb capped.pdb \
                                        --mae prep.mae \
                                        >     in.pdb

${PROCEDURE_DIR}/scripts/make_leapin.py --log serial.log \
                                        >     leap.in

tleap -s -f leap.in

${PROCEDURE_DIR}/scripts/acpype.py -p leap.parm7 \
                                   -x leap.rst7

modify_GROMACS_file leap_GMX.gro \
                    leap_GMX.top

mv leap_GMX.gro input.gro
mv leap_GMX.top input.top


rm -f serial.pdb \
      serial.log \
      prep.pdb   \
      prep.mae   \
      capped.pdb \
      in.pdb     \
      leap.in    \
      leap.log   \
      leap.pdb   \
      leap.parm7 \
      leap.rst7  \
      em.mdp     \
      md.mdp
