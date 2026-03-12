import os
import subprocess
import shutil
import sys
import re
import torch
import json

from concurrent.futures import ProcessPoolExecutor, as_completed



input_data_dir           = os.environ.get('INPUT_DATA_DIR')
diffdock_dir             = os.environ.get('DIFFDOCK_DIR')
    
diffdock_config_dir      = os.path.join(diffdock_dir, 'config')
pocket_dict_path         = os.path.join(diffdock_config_dir, 'pocket_dict.json')
diffdock_config_path     = os.path.join(diffdock_config_dir, 'diffdock_config.json')

with open(pocket_dict_path, 'r') as f:
    pocket_residues_dict = json.load(f)

with open(diffdock_config_path, 'r') as f:
    config = json.load(f)

bind_cliteria_distance   = config['bind_cliteria_distance__angstrom']
crash_cliteria_distance  = config['crash_cliteria_distance__angstrom']
gamma                    = config['gamma']
rho                      = config['rho']


def convert_and_move_files(output_dir, pattern_name, pocket_residues_dict=pocket_residues_dict):
    total_directories   = 0
    skipped_directories = 0
    max_length = max((len(d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))), default = 0)

    for sequence_dir in [d for d in os.listdir(output_dir) if d.startswith(pattern_name) and os.path.isdir(os.path.join(output_dir, d))]:
       sequence_path = os.path.join(output_dir, sequence_dir)
       if os.path.isdir(sequence_path):
           if not os.listdir(sequence_path):
               print(f"\033[93m[{sequence_dir:^{max_length}}]  *** Warning: Directory is empty, skipping. ***\033[0m")
               skipped_directories += 1
               continue

           total_directories += 1
           sdf_files = [os.path.join(sequence_path, f) for f in os.listdir(sequence_path) if f.endswith('.sdf') and "confidence" in f]
           sdf_file  = parallel_process_files(sdf_files, pocket_residues_dict)

           if sdf_file is None or not os.path.exists(sdf_file):
               print(f"\033[91m[{sequence_dir:^{max_length}}]  *** Warning: SDF file meeting criteria not found, skipping. ***\033[0m")
               skipped_directories += 1
               continue
           
           print(f"[{sequence_dir:^{max_length}}]  {os.path.basename(sdf_file)} is meeting the criteria.")

           pdb_file       = os.path.join(sequence_path, 'rank1.pdb')
           final_pdb_file = os.path.join(output_dir, f'{sequence_dir}.pdb')

           with open(os.devnull, 'w') as FNULL:
               subprocess.call(['obabel', sdf_file, '-O', pdb_file], stdout=FNULL, stderr=subprocess.STDOUT)

           shutil.copy(pdb_file, final_pdb_file)

    if total_directories > 0 and (skipped_directories / total_directories) > 0.9:
        print("\033[93mWarning: Over 90% of the directories were skipped due to empty or missing SDF files.\033[0m")


def is_sdf_file_valid(sdf_file):
    with open(sdf_file, 'r') as file:
        lines = file.readlines()
        atoms_section = False
        for i, line in enumerate(lines):
            if "V2000" in line or "V3000" in line:
                atoms_section = True
                continue
            if atoms_section:
                if not any(c.isalpha() for c in line):
                    atoms_section = False
                    continue
                parts = line.split()
                if len(parts) != 16:
                    return False
                try:
                    float(parts[0]), float(parts[1]), float(parts[2])                    
                except ValueError:
                    return False
    return True


def get_receptor_name(sdf_file_path):
    directory_path = os.path.dirname(sdf_file_path)
    directory_name = os.path.basename(directory_path)
    receptor_name  = directory_name.split('_')[0]
    return receptor_name


def make_receptor_pocket_position_tensor(receptor_name, pocket_residues_dict):
    if receptor_name not in pocket_residues_dict:
        return None
    coordinates = []
    pdb_file = os.path.join(input_data_dir, 'receptor', f'{receptor_name}.pdb')
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                residue_number = int(line[22:26].strip())
                if residue_number in pocket_residues_dict[receptor_name]:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coordinates.append([x, y, z])
    return torch.tensor(coordinates, dtype=torch.float32)


def make_all_receptor_position_tensor(receptor_name):
    coordinates = []
    pdb_file = os.path.join(input_data_dir, 'receptor', f'{receptor_name}.pdb')
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coordinates.append([x, y, z])
    return torch.tensor(coordinates, dtype=torch.float32)


def make_ligand_position_tensor(sdf_file):
    coordinates = []
    with open(sdf_file, 'r') as file:
        atoms_section = False
        for line in file:
            if "V2000" in line or "V3000" in line:
                atoms_section = True
                continue
            elif "M  END" in line:
                atoms_section = False
                break
            if atoms_section:
                parts = line.split()
                if len(parts) == 16:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coordinates.append([x, y, z])
                    except ValueError as e:
                        continue
    return torch.tensor(coordinates, dtype=torch.float32)


def compute_atom_distances(sdf_file, pocket_residues_dict):
    receptor_name               = get_receptor_name(sdf_file)
    receptor_positions          = make_receptor_pocket_position_tensor(receptor_name, pocket_residues_dict)
    ligand_positions            = make_ligand_position_tensor(sdf_file)

    if receptor_positions is None:
        return None

    receptor_positions_expanded = receptor_positions.unsqueeze(0)
    ligand_positions_expanded   = ligand_positions.unsqueeze(1)

    differences                 = receptor_positions_expanded - ligand_positions_expanded
    distances_tensor            = torch.sqrt((differences ** 2).sum(-1))
    return distances_tensor


def parse_counts_line(line):
    num_atoms = int(line[0:3])
    num_bonds = int(line[3:6]) 
    return num_atoms, num_bonds


def parse_bonds_from_sdf(sdf_file):
    bonds = []
    with open(sdf_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "V2000" in line or "V3000" in line:
                num_atoms, num_bonds = parse_counts_line(line)
                break

        atom_lines = lines[4             : num_atoms + 4            ]
        bond_lines = lines[num_atoms + 4 : num_atoms + 4 + num_bonds] 

        for line in bond_lines:
            atom1_str = line[0:3]
            atom2_str = line[3:6]
            if atom1_str.strip().isdigit() and atom2_str.strip().isdigit():
                atom1 = int(atom1_str) - 1 
                atom2 = int(atom2_str) - 1 
                bonds.append((atom1, atom2))
    return bonds


def set_infinite_distance_for_bonds(distances, bonds):
    for atom1, atom2 in bonds:
        distances[atom1, atom2] = distances[atom2, atom1] = float('inf')


def compute_self_distances(sdf_file):
    ligand_positions = make_ligand_position_tensor(sdf_file)
    differences      = ligand_positions.unsqueeze(1) - ligand_positions.unsqueeze(0)
    distances        = torch.sqrt((differences ** 2).sum(-1))
    mask             = torch.eye(distances.size(0), dtype=torch.bool)
    distances[mask]  = float('inf')

    bonds = parse_bonds_from_sdf(sdf_file)
    set_infinite_distance_for_bonds(distances, bonds)

    return distances


def detect_min_distance(distances_tensor):
    if distances_tensor is None:
        return None
    min_distance = torch.min(distances_tensor)
    return min_distance


def binding_check(min_distance, upper_limit=bind_cliteria_distance, lower_limit=crash_cliteria_distance):
    if min_distance is None:
        return True
    if min_distance < upper_limit and min_distance > lower_limit:
        return True
    else:
        return False


def intermolecurar_crash_check(min_distance, upper_limit=crash_cliteria_distance):
    if min_distance < upper_limit:
        return False
    else:
        return True


def compute_S(receptor_name, sdf_file, rho=rho):
    receptor_coords = make_all_receptor_position_tensor(receptor_name)
    ligand_coords   = make_ligand_position_tensor(sdf_file)
    diff            = receptor_coords.unsqueeze(1) - ligand_coords.unsqueeze(0)
    squared_dist    = (diff ** 2).sum(-1)
    S               = -rho * torch.log(torch.exp(-squared_dist / rho).sum(0))
    return S


def S_function_check(S, gamma=gamma):
    if torch.min(S) < gamma:
        return False
    else:
        return True


def choice_docking_structure(receptor_sequence_folder, pocket_residues_dict):
    sdf_files = [file for file in os.listdir(receptor_sequence_folder) if file.endswith(".sdf") and "confidence" in file]
    sdf_files.sort(key=lambda x: int(re.findall(r'rank(\d+)', x)[0]))

    for file in sdf_files:
        sdf_file = os.path.join(receptor_sequence_folder, file)
        if not is_sdf_file_valid(sdf_file):
            print(f"* Skipping invalid SDF file: {os.path.basename(sdf_file)}")
            continue 

        binding_distance              = detect_min_distance(compute_atom_distances(sdf_file, pocket_residues_dict))
        intermolecurar_crash_distance = detect_min_distance(compute_self_distances(sdf_file))
        S_values                      = compute_S(get_receptor_name(sdf_file), sdf_file)

        if binding_check(binding_distance) and intermolecurar_crash_check(intermolecurar_crash_distance) and S_function_check(S_values):    
            return sdf_file
    return None


def process_file(sdf_file, pocket_residues_dict):
    binding_distance              = detect_min_distance(compute_atom_distances(sdf_file, pocket_residues_dict))
    intermolecurar_crash_distance = detect_min_distance(compute_self_distances(sdf_file))
    S_values                      = compute_S(get_receptor_name(sdf_file), sdf_file)

    if binding_check(binding_distance) and intermolecurar_crash_check(intermolecurar_crash_distance) and S_function_check(S_values):
        return sdf_file
    else:
        return None


def parallel_process_files(sdf_files, pocket_residues_dict):
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, pocket_residues_dict): file for file in sdf_files}
        for future in as_completed(future_to_file):
            sdf_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    return result
            except Exception:
                pass
    return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_dir> <pattern_name>")
        sys.exit(1)

    output_dir   = sys.argv[1]
    pattern_name = sys.argv[2]
    
    rec_name = pattern_name.split("_")[0]
    if rec_name not in pocket_residues_dict:
        print(f'\033[93mWarning: {pattern_name} is not included in the pocket_residues_dict.\033[0m')
        print(f"\033[93m         Only checking for non-proximity of residues.\033[0m")

    convert_and_move_files(output_dir, pattern_name)
