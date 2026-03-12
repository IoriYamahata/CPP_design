
import os
import json
import argparse
import time
import sys
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from typing import Any, Tuple
from botorch.models.ensemble import EnsembleModel
from botorch.sampling.index_sampler import IndexSampler
from botorch.utils.multi_objective import is_non_dominated
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.optim.optimize import _split_batch_eval_acqf,_raise_deprecation_warning_if_kwargs

start = time.time() 



# ========== path and configure ========== #

input_data_dir   = os.environ.get('INPUT_DATA_DIR')
prediction_dir   = os.environ.get('PREDICTION_DIR')
optimize_dir     = os.environ.get('BAYESIAN_OPTIMIZATION_DIR')
cycle_result_dir = os.environ.get('CYCLE_RESULT_DIR')
cycle_num        = os.environ.get('CYCLE_NUMBER')
gpu_num          = os.environ.get('GPU_NUM')

next_cycle_num   = int(cycle_num) + 1

master_csv_path          = os.path.join(input_data_dir,   'master', 'master.csv')
next_cycle_sequence_path = os.path.join(input_data_dir,   'input_csv', f'cycle_{next_cycle_num}_input_sequence.csv')
input_config_path        = os.path.join(input_data_dir,   'config', 'input_data_config.json')
pred_config_path         = os.path.join(prediction_dir,   'config', 'prediction_config.json')
optimize_config_path     = os.path.join(optimize_dir,     'config', 'optimize_config.json')
master_csv_composed_path = os.path.join(cycle_result_dir, 'prediction_results', f'master_cycle_{cycle_num}_learn_input.csv')
candidates_save_path     = os.path.join(cycle_result_dir, 'optimize_results', f'{cycle_num}th_candites.csv')
acq_value_save_path      = os.path.join(cycle_result_dir, 'optimize_results', f'{cycle_num}th_acq_value.csv')

os.makedirs(os.path.dirname(candidates_save_path), exist_ok=True)



with open(input_config_path, 'r') as f:
    input_config    = json.load(f)

with open(pred_config_path, 'r') as f:
    pred_config = json.load(f)

with open(optimize_config_path, 'r') as f:
    optimize_config = json.load(f)



q                    = 100
max_batch_size       = optimize_config["max_batch_size"]
ref_point_list       = optimize_config["ref_point_list"]
prune_baseline       = optimize_config["prune_baseline"]
alpha                = optimize_config["alpha"]
max_iep              = optimize_config["max_iep"]
cache_root           = optimize_config["cache_root"]
unique               = optimize_config["unique"]
num_models           = pred_config["num_model"]


optimize_method_dict = {method['receptor'].strip(): method['optimization_method'] for method in optimize_config["optimize_method_list"]}

device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== Class and Function definition ========== #

def extract_pareto_indices_from_experimented_data(master_data: torch.Tensor, experiment_index: torch.Tensor) -> torch.Tensor:
    experiment_index       = torch.tensor(experiment_index, dtype=torch.long)
    experimented_data      = torch.index_select(master_data, 1, experiment_index.long())
    pareto_indices_set     = set()

    for model_idx in range(experimented_data.shape[0]):
        model_predictions  = experimented_data[model_idx, :, :]
        pareto_mask        = is_non_dominated(model_predictions)
        pareto_indices     = experiment_index[pareto_mask]

        pareto_indices_set.update(pareto_indices.tolist())

    combined_pareto_indices = torch.tensor(list(pareto_indices_set), dtype=torch.long)
    
    return combined_pareto_indices



class NeuralEnsemble(EnsembleModel):
    def __init__(self, master_data: Tensor):
        super().__init__()
        self.master_data = master_data

    def forward(self, X: Tensor) -> Tensor:
        indices = X[..., 0].long() 

        selected_predictions = master_data[:, indices, :]
        if selected_predictions.dim() == 4:
            selected_predictions = selected_predictions.permute(1, 0, 2, 3)

        return selected_predictions



def remake_n_experiment_index(n_experiment_index, candidate_list):
    choiced_indices        = [c.item() for c in torch.cat(candidate_list).squeeze(-2)]
    new_n_experiment_index = [idx for idx in n_experiment_index if idx not in choiced_indices]
    return new_n_experiment_index


def update_variable(q, candidates_list, n_experiment_index, master_data, experiment_index, modified_X_baseline):
    if len(candidates_list) == 0:
        new_choices = torch.tensor(n_experiment_index, dtype=float).unsqueeze(-1).to(device)
        return q, n_experiment_index, new_choices

    choiced_indices        = [int(c) for c in candidates_list]
    new_n_experiment_index = [idx for idx in n_experiment_index if idx not in choiced_indices]
    new_choices            = torch.tensor(new_n_experiment_index, dtype=float).unsqueeze(-1).to(device)

    if modified_X_baseline == True:
        
        master_data = master_data.to('cpu')

        pareto_experiment_index   = extract_pareto_indices_from_experimented_data(master_data, experiment_index)
        n_pareto_experiment_index = [idx for idx in experiment_index if not idx in pareto_experiment_index]
        new_X_baseline            = torch.tensor(n_pareto_experiment_index, dtype=float).unsqueeze(-1).to(device)
        
        print("")
        print(f'** pareto solutions {len(pareto_experiment_index)} data deleted from X_baseline **')
        print("")

        return new_choices, n_pareto_experiment_index, new_X_baseline, modified_X_baseline

    return new_choices, None, None, None



def candidates_and_acq_values_append_to_list(candidates_list, acq_value_list, temp_candidates, temp_acq_value, modified_X_baseline):
    if temp_candidates.nelement() > 0:
        modified_X_baseline = False
        if temp_candidates.nelement() == 1:
            candidates = [temp_candidates.item()]
        else:
            candidates = temp_candidates.squeeze(-1).tolist()
        for candidate in candidates:
            candidates_list.append(candidate)

        if temp_acq_value.nelement() == 1:
            acq_values = [temp_acq_value.item()]
        else:
            acq_values = temp_acq_value.squeeze(-1).tolist()
        for acq_value in acq_values:
            acq_value_list.append(acq_value)

    else:
        if modified_X_baseline:
            print("random choice")

            num_random_samples = min(q-len(candidates_list), remaining_choices.size(0))
            random_indices     = torch.randint(0, remaining_choices.size(0), (num_random_samples,))
            random_candidates  = remaining_choices[random_indices].squeeze(-1)

            for candidate in random_candidates.tolist():
                candidates_list.append(candidate)
            for _ in range(num_random_samples):
                acq_value_list.append(0) 
        else:
            modified_X_baseline = True

    return modified_X_baseline


def optimize_acqf_discrete(
    acq_function:   AcquisitionFunction,
    q:              int,
    choices:        Tensor,
    max_batch_size: int  = 2048,
    unique:         bool = True,
    **kwargs:       Any,
) -> Tuple[Tensor, Tensor]:

    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "Discrete optimization is not supported for"
            "one-shot acquisition functions."
        )
    if choices.numel() == 0:
        raise InputDataError("`choices` must be non-emtpy.")
    
    _raise_deprecation_warning_if_kwargs("optimize_acqf_discrete", kwargs)
    choices_batched = choices.unsqueeze(-2)

    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for i in range(q):
            start_for = time.time()

            with torch.no_grad():
                start_acq  = time.time()
                acq_values = _split_batch_eval_acqf(
                    acq_function   = acq_function,
                    X              = choices_batched,
                    max_batch_size = max_batch_size,
                ) 
                end_acq    = time.time()

                if (acq_values == 0).all():
                    break                 

            best_idx = torch.argmax(acq_values)
            candidate_list.append(choices_batched[best_idx])
            acq_value_list.append(acq_values[best_idx])
            
            candidates = torch.cat(candidate_list, dim=-2)

            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            
            if unique:
                choices_batched = torch.cat(
                    [choices_batched[:best_idx], choices_batched[best_idx + 1 :]]
                )
            end_for   = time.time() 
            time_acq  = end_acq - start_acq
            time_diff = end_for - start_for
            print(f"Iteration {i+1}/{q} completed. | Acquisition Time = {time_acq:.2f} seconds| Time taken = {time_diff:.2f} seconds")

        acq_function.set_X_pending(base_X_pending)

        if all(value == 0 for value in acq_value_list):
            return torch.empty(0, 1), torch.empty(0)

        return candidates, torch.stack(acq_value_list)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function    = acq_function, 
            X               = choices_batched, 
            max_batch_size  = max_batch_size
        )

    best_idx = torch.argmax(acq_values)
    return choices_batched[best_idx], acq_values[best_idx]



# ========== Create master_data, X_baseline, choices ========== #

master_all_data = pd.read_csv(master_csv_composed_path)

master_all_data.index.name = "Index"
master_all_data.reset_index(inplace=True)

## master_data 
mean_columns = [col for col in master_all_data.columns if "mean_" in col]
extracted_dataset = master_all_data[mean_columns]

receptors = [method['receptor'].strip() for method in optimize_config["optimize_method_list"]]

tensors   = []
for receptor, optimize_method in optimize_method_dict.items():
    if optimize_method not in ['minimize', 'maximize']:
        raise ValueError(f"Invalid optimization method for receptor {receptor}: {optimize_method}. Please use 'maximize' or 'minimize'.")

    filtered_data = extracted_dataset.filter(like=receptor).values
    tensor        = torch.tensor(filtered_data, dtype=torch.float32)

    if optimize_method == 'minimize':
        print(f"Minimize for {receptor}")
        tensor = - tensor

    elif optimize_method == 'maximize':
        print(f"Maximize for {receptor}")
        tensor =   tensor

    tensors.append(tensor)

master_data        = torch.stack(tensors, dim=0).permute(2, 1, 0)


## index
experiment_index   = master_all_data["Index"].tolist()
n_experiment_index = master_all_data["Index"].tolist()

## X_baseline, choices
X_baseline = torch.empty((0, 1), dtype=float)
choices    = torch.tensor(experiment_index, dtype=float).unsqueeze(-1)

print(f"Number of X_baseline: {X_baseline.size()}")
print(f"Number of choices: {choices.size()}")


# ========== Bayesian Optimization ========== #

## set sampler and ensemble model
master_data = master_data.to(device)
choices     = choices.to(device) 
X_baseline  = X_baseline.to(device)

sampler = IndexSampler(sample_shape=(num_models))
sampler.base_samples = torch.arange(num_models)
model = NeuralEnsemble(master_data=master_data)

ref_point = torch.tensor(ref_point_list).to(device)

candidates_list   = []
acq_value_list    = []
remaining_choices = choices.clone()
loop_num = 1

while len(candidates_list) < q and remaining_choices.size(0) > 0:
    modified_X_baseline_flag = False
    
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model           = model,
        ref_point       = ref_point,
        X_baseline      = X_baseline,
        sampler         = sampler,
        prune_baseline  = False,
        alpha           = alpha,
        max_iep         = max_iep,
        cache_root      = cache_root
        )
    
    temp_candidates, temp_acq_value = optimize_acqf_discrete(
        acq_function    = acq_func,
        q               = min(q-len(candidates_list), remaining_choices.size(0)),
        choices         = remaining_choices,
        max_batch_size  = max_batch_size,
        unique          = unique
        )
    
    modified_X_baseline_flag = candidates_and_acq_values_append_to_list(
        candidates_list      = candidates_list, 
        acq_value_list       = acq_value_list, 
        temp_candidates      = temp_candidates, 
        temp_acq_value       = temp_acq_value, 
        modified_X_baseline  = modified_X_baseline_flag
        )

    if len(candidates_list) < q:
        print(f"[Loop {loop_num}] Only got {len(temp_candidates.squeeze(-1))} candidates. Loop {loop_num+1} started")

        remaining_choices, experiment_index_update, X_baseline_update, modified_X_baseline_flag_update \
            = update_variable(
                q                    = q, 
                candidates_list      = candidates_list, 
                n_experiment_index   = n_experiment_index, 
                master_data          = master_data, 
                experiment_index     = experiment_index, 
                modified_X_baseline  = modified_X_baseline_flag
                )

        if experiment_index_update is not None:
            experiment_index         = experiment_index_update
            X_baseline               = X_baseline_update
            modified_X_baseline_flag = modified_X_baseline_flag_update
        
        loop_num += 1




# ========== Save results ========== #
candidates_tensor = torch.tensor(candidates_list)
acq_value_tensor  = torch.tensor(acq_value_list)

candidates_np     = candidates_tensor.cpu().numpy()
acq_value_np      = acq_value_tensor.cpu().numpy()

df_candidates     = pd.DataFrame(candidates_np, columns=['candidates Index'])
df_acq_value      = pd.DataFrame(acq_value_np,  columns=['Acquisition Value'])

df_candidates.to_csv(candidates_save_path, index=False)
df_acq_value.to_csv(acq_value_save_path, index=False)


# ========== make input file of next cycle ========== #

candidates_indices = candidates_np.astype(int).flatten() 
master_file        = pd.read_csv(master_csv_path)
selected_data      = master_file.iloc[candidates_indices]

selected_data.to_csv(next_cycle_sequence_path, index=False)

end = time.time() 
time = end - start

print(f"Optimization is completed. Total time taken = {time:.1f} seconds")
