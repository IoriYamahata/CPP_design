import os
import csv
import glob
import sys
import json


input_data_dir             = os.environ.get('INPUT_DATA_DIR')
prediction_dir             = os.environ.get('PREDICTION_DIR')

input_config_path          = os.path.join(input_data_dir, 'config', 'input_data_config.json')
prediction_config_path     = os.path.join(prediction_dir, 'config', 'prediction_config.json')

energy_num                 = 11

with open(prediction_config_path, 'r') as f:
    config = json.load(f)
    invalid_energy_threshold = config['invalid_energy_threshold']


def load_config():
    with open(input_config_path, 'r') as file:
        config = json.load(file)
    return config['target_receptor_list']


def find_energy_files(data_dir):
    return glob.glob(os.path.join(data_dir, "*/energy.csv"))


def get_receptor_and_sequence(energy_file_path):
    directory_path          = os.path.dirname(energy_file_path)
    folder_name             = os.path.basename(directory_path)
    receptor_name, sequence = folder_name.split("_", 1)
    return receptor_name, sequence


def get_energy(energy_file):
    energies = []
    with open(energy_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if "Delta Energy Terms" in row:
                for _ in range(1): 
                    next(reader)
                for j, data in enumerate(reader):
                    if j == energy_num : break
                    energies.append(str(-float(data[-1])))  # 符号を反転
                break
    return energies


def residue_wise_normalize_energy(sequence, energies):
    residue_count       = len(sequence)
    normalized_energies = [float(energy) / residue_count for energy in energies]
    return normalized_energies


def update_sequence_energy_dict(energy_file, sequence_energy_dict, receptors):
    receptor, sequence = get_receptor_and_sequence(energy_file)
    energies           = get_energy(energy_file)
    sequence_key       = f"{sequence}"

    if sequence_key not in sequence_energy_dict:
        sequence_energy_dict[sequence_key] = {f"{rec}_energy": [None] * energy_num for rec in receptors}
    
    if receptor in receptors:
        sequence_energy_dict[sequence_key][f"{receptor}_energy"] = energies


def create_sequence_energy_dict(data_dir, receptors):
    sequence_energy_dict = {}
    energy_files         = find_energy_files(data_dir)

    for energy_file in energy_files:
        update_sequence_energy_dict(energy_file, sequence_energy_dict, receptors)
    return sequence_energy_dict


def remove_uncomplete_sequences(sequence_energy_dict, receptors):
    preprocess_number = len(sequence_energy_dict)
    incomplete_sequences = []
    crash_sequences = []

    for sequence in list(sequence_energy_dict.keys()):
        if any(None in sequence_energy_dict[sequence][f"{receptor}_energy"] for receptor in receptors):
            incomplete_sequences.append(sequence)
            del sequence_energy_dict[sequence]

    for sequence in list(sequence_energy_dict.keys()):
        if any(float(energy) < (-1 * invalid_energy_threshold) for receptor in receptors for energy in sequence_energy_dict[sequence][f"{receptor}_energy"]):
            crash_sequences.append(sequence)

    for sequence in crash_sequences:
        del sequence_energy_dict[sequence]

    for sequence in sequence_energy_dict.keys():
        for receptor in receptors:
            sequence_energy_dict[sequence][f"{receptor}_energy"] = residue_wise_normalize_energy(sequence, sequence_energy_dict[sequence][f"{receptor}_energy"])

    if incomplete_sequences:
        print(f"\n\033[93mPreprocessing Warning - unpaired sequence\033[0m")
        print(f"  There are sequences that do not have energy values for all required receptors.")
        print(f"  Sequences:")
        for seq in incomplete_sequences:
            print(f"   [{seq}]")

    if crash_sequences:
        print(f"\n\033[93mPreprocessing Warning - critical low energy sequence\033[0m")
        print(f"  There are energy values larger than {invalid_energy_threshold} kcal/mol.")
        print(f"  Sequences:\n   [" + "]\n   [".join(crash_sequences) + "]")

    print(f"\n\033[92mPreprocessing Summary\033[0m")
    print(f" Signs inverted and filtering completed.")
    print(f"  Total number of sequences: {preprocess_number}")
    print(f"  Sequences are left       : {len(sequence_energy_dict)} ")
    print(f"  Sequences are removed    : {preprocess_number - len(sequence_energy_dict)} ")

    return sequence_energy_dict



def write_sequence_energy_dict(sequence_energy_dict, output_file, receptors):
    header = ["Sequence"] + [f"{rec}_energy_{i}" for rec in receptors for i in range(1, energy_num + 1)]
    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for sequence, energies in sequence_energy_dict.items():
            row = [sequence] + [energy for rec in receptors for energy in energies[f"{rec}_energy"]]
            writer.writerow(row)

def main(data_dir, output_file):
    receptors = load_config()
    sequence_energy_dict = create_sequence_energy_dict(data_dir, receptors)
    sequence_energy_dict = remove_uncomplete_sequences(sequence_energy_dict, receptors)
    write_sequence_energy_dict(sequence_energy_dict, output_file, receptors)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))


