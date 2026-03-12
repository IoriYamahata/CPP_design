import os
import sys
import argparse


input_data_dir = os.environ.get('INPUT_DATA_DIR')
receptor_dir   = os.path.join(input_data_dir, 'receptor')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gro'                    ,
                        type     = str             ,
                        help     = 'input gro file',
                        required = True            ,)
    return parser

def get_receptor_name(gro_file_path):
    directory_path    = os.path.dirname(gro_file_path)
    folder_name       = os.path.basename(directory_path)
    receptor_name     = folder_name.split("_")[0]
    return receptor_name


def get_last_residue_number(gro_file_path):
    with open(gro_file_path, 'r') as gro_file:
        lines          = gro_file.readlines()
        last_line      = lines[-2]
        residue_number = int(last_line[0:5].strip())
    return residue_number   


def count_residues(receptor_name):
    residue_set = set()
    with open(f"{receptor_dir}/{receptor_name}.pdb", 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                residue_id = (line[17:20].strip(), line[22:26].strip(), line[21].strip())
                residue_set.add(residue_id)
    return len(residue_set)


def make_ndx_command_string(receptor_name, last_residue_number):
    receptor_end_residue = count_residues(receptor_name) + 2
    ndx_command_string   = f"r 1-{receptor_end_residue}\n"
    ndx_command_string  += f"name 10 Receptor\n"
    ndx_command_string  += f"r {receptor_end_residue+1}-{last_residue_number}\n"
    ndx_command_string  += f"name 11 Ligand\n"
    ndx_command_string  += f"q\n"
    return ndx_command_string


def main():
    parser = get_parser()
    args = parser.parse_args()

    receptor_name       = get_receptor_name(args.gro)
    last_residue_number = get_last_residue_number(args.gro)
    ndx_command_string  = make_ndx_command_string(receptor_name, last_residue_number)
    with open("commands.in", "w") as file:
        file.write(ndx_command_string)


if __name__ == '__main__':
    sys.exit(main())
