import json
import math
import os

import pandas as pd

diffdock_dir      = os.environ.get('DIFFDOCK_DIR')
input_data_dir    = os.environ.get('INPUT_DATA_DIR')

directory_path    = diffdock_dir+'/DiffDock/data'

csv_files         = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('_tmp.csv')]
combined_csv      = pd.concat([pd.read_csv(f) for f in csv_files])

combined_csv.to_csv(os.path.join(directory_path, 'diffdock_input.csv'), index=False)

config_path = os.path.join(diffdock_dir, 'config', 'diffdock_config.json')
if os.environ.get('DOCKING_NUMBER_PER_GPU'):
    num_per_gpu = int(os.environ.get('DOCKING_NUMBER_PER_GPU'))
else:
    with open(config_path, 'r') as f:
        num_per_gpu = int(json.load(f)['docking_number_per_GPU'])
rows_per_file     = num_per_gpu


combined_csv_path = os.path.join(directory_path, 'diffdock_input.csv')
combined_csv      = pd.read_csv(combined_csv_path)
total_rows        = len(combined_csv)

file_number       = math.ceil(total_rows / rows_per_file)
output_files      = [pd.DataFrame() for _ in range(file_number)]

for i, row in combined_csv.iterrows():
    file_index = i % file_number
    output_files[file_index] = pd.concat([output_files[file_index], pd.DataFrame([row])], ignore_index=True)

for i, df in enumerate(output_files, 1):
    if not df.empty:
        df.to_csv(os.path.join(directory_path, f'{i}_diffdock_input.csv'), index=False)
