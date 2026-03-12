import os
import pickle
import numpy as np
import pandas as pd
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch              import nn
from torch.utils.data   import DataLoader, random_split
from torch.optim        import Adam

from convolution_model  import two_block_convolution_model, number_encorded_dataset, train_epoch, valid_epoch, plot_and_save



# ========== path and configure ========== #
# parser
parser = argparse.ArgumentParser()
parser.add_argument("model_id"   , type = str, help = "model id.")
parser.add_argument("training_id", type = int, help = "training id.")
args = parser.parse_args()

# environment variables
input_data_dir             = os.environ.get('INPUT_DATA_DIR')
project_dir                = os.environ.get('PROJECT_DIR')
prediction_dir             = os.environ.get('PREDICTION_DIR')
cycle_result_dir           = os.environ.get('CYCLE_RESULT_DIR')
cycle_number               = os.environ.get('CYCLE_NUMBER')

# file_path
input_config_path          = os.path.join(input_data_dir,   "config", f"input_data_config.json") 
prediction_config_path     = os.path.join(prediction_dir,   "config", f"prediction_config.json") 
model_config_path          = os.path.join(prediction_dir,   "config", f"model_config_{args.model_id}.json")
amino_acid_dictionary_path = os.path.join(prediction_dir,   "config", f"amino_acid_dictionary.json")
training_data_file_path    = os.path.join(cycle_result_dir, "training_dataset.csv")
embedding_data_file_path   = os.path.join(cycle_result_dir, "embedding_mask_dataset.pkl")
train_save_path            = os.path.join(cycle_result_dir, "training_results", f"{args.model_id}-{args.training_id}_Train.png")
valid_save_path            = os.path.join(cycle_result_dir, "training_results", f"{args.model_id}-{args.training_id}_Valid.png")
loss_save_path             = os.path.join(cycle_result_dir, "training_results", f"{args.model_id}-{args.training_id}_Loss.png")
model_save_path            = os.path.join(cycle_result_dir, "training_results", f"{args.model_id}-{args.training_id}_state_dict.pt")
         

with open(model_config_path, 'r') as f:
    model_config = json.load(f)

with open(input_config_path, 'r') as f:
    input_config = json.load(f)

with open(prediction_config_path, 'r') as f:
    prediction_config = json.load(f)

with open(amino_acid_dictionary_path, 'r') as f:
    amino_acid_dictionary = json.load(f)

# config
batch_size                 = model_config['batch_size']
learning_rate              = model_config['learning_rate']
num_epochs                 = model_config['num_epochs']
max_seq_length             = model_config['max_seq_length']
patience                   = model_config['patience']
prediction_batch_size      = prediction_config['prediction_batch_size']
target_receptor_list       = input_config['target_receptor_list']
receptor_num               = len(target_receptor_list)
energy_num                 = 11

# device
device                     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== function definition ========== #
def number_encording(sequences, amino_acid_dictionary, max_seq_length):
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequence = [amino_acid_dictionary.get(aa, amino_acid_dictionary['PAD']) for aa in sequence]
        padded_sequence  = encoded_sequence[:max_seq_length] + [amino_acid_dictionary['PAD']] * (max_seq_length - len(encoded_sequence))
        encoded_sequences.append(padded_sequence)
    
    encoded_tensor = torch.tensor(encoded_sequences, dtype=torch.long)
    return encoded_tensor


# ========== data preparetion ========== #
# dataset, dataloader
data         = pd.read_csv(training_data_file_path)
targets      = data.drop(columns = ['Sequence']).values
sequence     = number_encording(data['Sequence'].values, amino_acid_dictionary, max_seq_length)

dataset      = number_encorded_dataset(sequence, targets)
dataset_size = len(dataset)

valid_size   = int(dataset_size * 0.1)
train_size   = dataset_size - valid_size

# Ensure non-empty splits for small datasets
if dataset_size < 2:
    train_dataset = dataset
    valid_dataset = None
elif valid_size < 1:
    valid_size = 1
    train_size = dataset_size - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
else:
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) if valid_dataset is not None else None


# ========== model training ========== #

model        = two_block_convolution_model(amino_acid_dictionary = amino_acid_dictionary, 
                                           config                = model_config, 
                                           max_seq_length        = max_seq_length,
                                           receptor_num          = receptor_num
                                           )

print(f"[{args.model_id}-{args.training_id}] Model Summary:")
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion    = nn.GaussianNLLLoss() 
optimizer    = Adam(model.parameters(), lr = learning_rate,  weight_decay = 1e-5)


train_losses = []
valid_losses = []

train_actuals     = {receptor: [] for receptor in target_receptor_list}
valid_actuals     = {receptor: [] for receptor in target_receptor_list}
train_predictions = {receptor: [] for receptor in target_receptor_list}
valid_predictions = {receptor: [] for receptor in target_receptor_list}

best_val_loss    = float("inf")
no_improve_epoch = 0

model.to(device)
for epoch in range(num_epochs):
    train_loss, train_results = train_epoch(model, train_loader, criterion, optimizer, device, target_receptor_list, energy_num)
    if valid_loader is not None:
        valid_loss, valid_results = valid_epoch(model, valid_loader, criterion, device, target_receptor_list, energy_num)
    else:
        valid_loss, valid_results = train_loss, {r: {"actuals": [], "predictions": []} for r in target_receptor_list}
    
    final_train_actuals     = {receptor: train_results[receptor]['actuals']     for receptor in target_receptor_list}
    final_valid_actuals     = {receptor: valid_results[receptor]['actuals']     for receptor in target_receptor_list}
    final_train_predictions = {receptor: train_results[receptor]['predictions'] for receptor in target_receptor_list}
    final_valid_predictions = {receptor: valid_results[receptor]['predictions'] for receptor in target_receptor_list}

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    if valid_loss < best_val_loss:
        best_val_loss       = valid_loss
        no_improve_epoch    = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        no_improve_epoch   += 1

    print(f"{args.model_id}-{args.training_id} Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loader is not None and no_improve_epoch  >= patience:
        print(f"No improvement in the last {patience} epochs. Stopping training.")
        break

# ========== save results ========== #

os.makedirs(os.path.dirname(valid_save_path), exist_ok=True)

plot_and_save(final_train_actuals, final_train_predictions, train_save_path)
if valid_loader is not None:
    plot_and_save(final_valid_actuals, final_valid_predictions, valid_save_path)

## loss plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label = 'Train Loss')
if valid_loader is not None:
    plt.plot(valid_losses, label = 'Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(loss_save_path)


## ========== prediction ========== ##

master_csv_path = os.path.join(input_data_dir, 'master', 'master.csv')
temp_csv_path   = os.path.join(cycle_result_dir, 'prediction_results', f"master_cycle_{cycle_number}.csv.{args.model_id}-{args.training_id}")

master_data     = pd.read_csv(master_csv_path)
master_sequence = number_encording(master_data['Sequence'].values, amino_acid_dictionary, max_seq_length)

total_batches   = len(master_sequence) // prediction_batch_size 

model.eval()

predictions     = [] 
prediction_data = {}

for i in range(0, len(master_sequence), prediction_batch_size):
    sequence_batch  = master_sequence[i:i+prediction_batch_size].clone().detach().to(device)
    with torch.no_grad():
        outputs     = model(sequence_batch)
        outputs_cpu = outputs.cpu()
    predictions.append(outputs_cpu)
    print(f"{args.model_id}-{args.training_id} Batch {i//prediction_batch_size+1}/{total_batches} processed.")

predictions_tensor = torch.cat(predictions, dim=1) 

for i, receptor in enumerate(target_receptor_list):
    means = predictions_tensor[i, :, 0].numpy()
    vars  = predictions_tensor[i, :, 1].numpy()

    prediction_data[f"{receptor}_prediction_mean_{args.model_id}-{args.training_id}"] = means.flatten()
    prediction_data[f"{receptor}_prediction_var_{ args.model_id}-{args.training_id}"] = vars.flatten()


predictions_df = pd.DataFrame(prediction_data)
predictions_df.to_csv(temp_csv_path, index=False)
