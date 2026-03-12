import os
import csv
import sys

diffdock_dir   = os.environ.get('DIFFDOCK_DIR')
input_data_dir = os.environ.get('INPUT_DATA_DIR')



def create_csv(sequence_directory, protein_name, output_dir):
    csv_file_path = os.path.join(output_dir, f'{protein_name}_tmp.csv')
    protein_path  = os.path.join(input_data_dir, 'receptor', f'{protein_name}.pdb')

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['complex_name', 'protein_path', 'ligand_description', 'protein_sequence'])

        for filename in os.listdir(sequence_directory):
            if filename.endswith('.sdf'):
                sequence_name      = filename.replace('.sdf', '')
                complex_name       = f'{protein_name}_{sequence_name}' 
                ligand_description = os.path.join(sequence_directory, filename)
                writer.writerow([complex_name, protein_path, ligand_description, ''])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_input_csv.py path/to/sequence_directory protein_name")
        sys.exit(1)

    sequence_directory = sys.argv[1]
    protein_name       = sys.argv[2]
    output_directory   = diffdock_dir + '/DiffDock/data'

    create_csv(sequence_directory, protein_name, output_directory)
