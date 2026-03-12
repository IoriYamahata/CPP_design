import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def sequence_to_mol(sequence):
    mol = Chem.MolFromSequence(sequence)
    
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, (0.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    
    return mol

def mol_to_sdf(mol, output_dir, output_file):
    sdf_file_path = os.path.join(output_dir, output_file)
    writer = Chem.SDWriter(sdf_file_path)
    writer.write(mol)
    writer.close()

def run_to_sdf_on_sequences(csv_file_path, output_dir):
    df = pd.read_csv(csv_file_path)

    for sequence in df['Sequence']:
        mol = sequence_to_mol(sequence)
        sdf_file_name = f"{sequence}.sdf"
        mol_to_sdf(mol, output_dir, sdf_file_name)

def main():
    parser = argparse.ArgumentParser(description='Convert amino acid sequences in a CSV file to SDF files.')
    parser.add_argument('csv_file_path', type=str, help='Path to the CSV file containing sequences.')
    parser.add_argument('output_dir', type=str, help='Directory to store output SDF files.')

    args = parser.parse_args()
    run_to_sdf_on_sequences(args.csv_file_path, args.output_dir)

if __name__ == "__main__":
    main()

