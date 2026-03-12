import sys
import os

def combine_pdb_files(protein_file, ligand_file, output_file):
    combined_pdb_content = []

    with open(protein_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                combined_pdb_content.append(line)

    with open(ligand_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                new_line = line[:21] + 'B' + line[22:]
                combined_pdb_content.append(new_line)

    with open(output_file, 'w') as file:
        file.writelines(combined_pdb_content)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_pdb.py protein.pdb ligand.pdb output.pdb")
        sys.exit(1)

    protein_pdb = sys.argv[1]
    ligand_pdb  = sys.argv[2]
    output_pdb  = sys.argv[3]

    if not os.path.isfile(protein_pdb) or not os.path.isfile(ligand_pdb):
        print("Error: Protein or ligand file does not exist.")
        sys.exit(1)

    combine_pdb_files(protein_pdb, ligand_pdb, output_pdb)

