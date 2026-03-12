import os
import re
import sys
from tqdm import tqdm

def sort_pdb_file(file_path):
    atom_records = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_records.append(line)
    atom_records.sort(key = lambda x: int(re.findall(r'\d+', x[22:26])[0]))
    return atom_records


def pdb_residue_length(atom_records):
    residue_numbers = [int(re.findall(r'\d+', atom[22:26])[0]) for atom in atom_records if re.findall(r'\d+', atom[22:26])]
    max_residue_number = max(residue_numbers, default=0)
    return max_residue_number


def get_file_name(file_path):
    return os.path.basename(file_path)


def get_sequence(file_name):
    return file_name.split('_')[-1].split('.')[0]


def filename_residue_length(sequence):
    return len(sequence)

def edit_pdb_for_ser(file_path):
    atom_order     = ['N ', 'CA', 'C ', 'O ', 'CB', 'OG', 'O ']
    original_order = ['N ', 'C ', 'C ', 'O ', 'C ', 'O ', 'O ']
    modified_lines = []
    atom_records   = []
    atom_index     = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("HETATM") and "UNK" in line[17:20]:
                line = line.replace("HETATM", "ATOM  ")
                line = line[:17] + "SER" + line[20:]
                current_atom = line[13:16].strip()
                if atom_index < len(original_order) and f"{current_atom} " == original_order[atom_index]:
                    new_atom = atom_order[atom_index]
                    line     = line[:13] + new_atom.ljust(3) + line[16:]
                    atom_index += 1

            modified_lines.append(line)
            if line.startswith("ATOM"):
                atom_records.append(line)

    with open(file_path, 'w') as file:
        for line in modified_lines:
            file.write(line)
    
    atom_records.sort(key=lambda x: int(re.findall(r'\d+', x[22:26])[0]))
    return atom_records



def create_sorted_pdb_file(original_file_path, sorted_atoms, output_dir):
    original_file_name = os.path.basename(original_file_path)
    sorted_file_path   = os.path.join(output_dir, original_file_name)

    with open(original_file_path, 'r') as original_file, \
         open(sorted_file_path,   'w') as sorted_file:
        
        for line in original_file:
            if not line.startswith("ATOM"):
                sorted_file.write(line)
        for atom_line in sorted_atoms:
            sorted_file.write(atom_line)

    return sorted_file_path


def main(process_dir):
    file_list = os.listdir(process_dir)
    max_length = max((len(d) for d in file_list))

    for file_name in tqdm(file_list, desc="Processing files", unit="file"):
        if file_name.endswith('.pdb'):
            file_path          = os.path.join(process_dir, file_name)

            sequence_name      = get_sequence(file_name)
            sequence_length    = filename_residue_length(sequence_name)

            if sequence_name.endswith("S"):
                sorted_atoms   = edit_pdb_for_ser(file_path)
            else:
                sorted_atoms   = sort_pdb_file(file_path)
            
            max_residue_number = pdb_residue_length(sorted_atoms)



            if max_residue_number == sequence_length:
                print(f'[{file_name:^{max_length}}] is matching the sequence length of {sequence_length}')
                create_sorted_pdb_file(file_path, sorted_atoms, process_dir)

            else:
                print(f'\033[91m[{file_name:^{max_length}}]  is not matching the sequence length of {sequence_length}\033[0m')
                os.remove(file_path) 

if __name__ == "__main__":
    process_dir = sys.argv[1]
    main(process_dir)

