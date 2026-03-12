import os
import json
import pandas as pd


input_data_dir     = os.environ.get('INPUT_DATA_DIR')
optimize_dir       = os.environ.get('BAYESIAN_OPTIMIZATION_DIR')
cycle_result_dir   = os.environ.get('CYCLE_RESULT_DIR')
cycle_num          = os.environ.get('CYCLE_NUMBER')

sequence_master    = os.path.join(input_data_dir,   'master', 'master.csv')
config_path        = os.path.join(optimize_dir,     'config', 'optimize_config.json')
training_data_path = os.path.join(cycle_result_dir, 'training_dataset.csv')
master_csv_path    = os.path.join(cycle_result_dir, 'prediction_results', f'master_cycle_{cycle_num}.csv')
master_csv_out     = os.path.join(cycle_result_dir, 'prediction_results', f'master_cycle_{cycle_num}_learn_input.csv')
output_path        = os.path.join(cycle_result_dir, 'pareto_solutions.csv')

energy_num         = 11 #100ps simulation 10steps


with open(config_path, 'r') as f:
    optimize_config  = json.load(f)

optimize_method_dict = \
    {method['receptor'].strip(): method['optimization_method'] for method in optimize_config["optimize_method_list"]}

tested = pd.read_csv(training_data_path)
master = pd.read_csv(master_csv_path)

for receptor, _ in optimize_method_dict.items():
    tested[f'test_{receptor}_mean'] = tested[[f'{receptor}_energy_{i}' for i in range(1, energy_num+1)]].mean(axis=1)
    master[f'test_{receptor}_mean'] = None

tested['test'] = True
master['test'] = False

tested = tested.set_index('Sequence')
master = master.set_index('Sequence')

update = ['test'] + [f'test_{receptor}_mean' for receptor in optimize_method_dict.keys()]
tested_unique = tested.loc[~tested.index.duplicated(keep='first')]


master.update(tested_unique[update])
master.reset_index().to_csv(master_csv_out, index=False)

print("Identification has completed.")
