
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats        import pearsonr
from torch            import nn
from torch.utils.data import Dataset



class number_encorded_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        sequence = self.data[index]
        label = self.label[index]
        return sequence, label

    def __len__(self):
        return len(self.data)


class two_block_convolution_model(nn.Module):
    def __init__(self, amino_acid_dictionary, config, max_seq_length, receptor_num):
        super(two_block_convolution_model, self).__init__()
        self.receptor_num       = receptor_num

        num_amino_acids         = len(amino_acid_dictionary)
        num_convolution_layers  = config['num_convolution_layers']
        layer_hidden_dim        = config['layer_hidden_dim']

        layer1_kernel_size      = config['layer1_kernel_size']
        layer1_dilation         = config['layer1_dilation']
        layer1_padding          = (layer1_dilation * (layer1_kernel_size - 1)) // 2
        
        layer2_kernel_size      = config['layer2_kernel_size']
        layer2_dilation         = config['layer2_dilation']
        layer2_padding          = (layer2_dilation * (layer2_kernel_size - 1)) // 2

        conv_layers = []
        for _ in range(num_convolution_layers):

            conv_layers += [
                nn.Conv1d(in_channels  = layer_hidden_dim,  
                          out_channels = layer_hidden_dim, 
                          kernel_size  = layer1_kernel_size, 
                          padding      = layer1_padding, 
                          dilation     = layer1_dilation),
                nn.BatchNorm1d(layer_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_prob'])
            ]

            conv_layers += [
                nn.Conv1d(in_channels  = layer_hidden_dim, 
                          out_channels = layer_hidden_dim, 
                          kernel_size  = layer2_kernel_size, 
                          padding      = layer2_padding, 
                          dilation     = layer2_dilation),
                nn.BatchNorm1d(layer_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_prob'])
            ]

        self.embedding   = nn.Embedding(num_amino_acids, layer_hidden_dim, padding_idx = 0)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc          = nn.Linear(layer_hidden_dim * max_seq_length, receptor_num * 2) 
        

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        means = []
        vars  = []
        
        for i in range(self.receptor_num):
            mean = x[:, i * 2]
            var  = x[:, i * 2 + 1]
            var  = F.softplus(var)
            means.append(mean.unsqueeze(1))
            vars.append(  var.unsqueeze(1))
        
        means   = torch.stack(means, dim=0) 
        vars    = torch.stack(vars,  dim=0)         
        outputs = torch.cat([means, vars], dim=2) # outputs shape: (receptor_num, batch_size, [mean, var])
        
        return outputs




def train_epoch(model, train_loader, criterion, optimizer, device, target_receptor_list, energy_num):
    model.train()
    total_loss = 0
    results = {receptor: {'actuals': [], 'predictions': []} for receptor in target_receptor_list}
    

    for sequence, target in train_loader:
        optimizer.zero_grad()
        sequence = sequence.to(device)
        target   = target.to(device)

        outputs = model(sequence)
        loss = 0
        
        for i, receptor in enumerate(target_receptor_list):
            start_index      = energy_num * i
            end_index        = start_index + energy_num
            target_receptor  = target[:, start_index:end_index]

            receptor_mean    = outputs[i, :, 0] 
            receptor_var     = outputs[i, :, 1] 

            receptor_loss    = criterion(receptor_mean.unsqueeze(-1).expand_as(target_receptor), 
                                         target_receptor, 
                                         receptor_var.unsqueeze(-1).expand_as(target_receptor))
            loss            += receptor_loss

            results[receptor]['actuals'].append(target_receptor.cpu().numpy().tolist()) 
            results[receptor]['predictions'].append(receptor_mean.detach().cpu().numpy().tolist())


        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss, results


def valid_epoch(model, valid_loader, criterion, device, target_receptor_list, energy_num):
    model.eval()
    total_loss = 0
    results = {receptor: {'actuals': [], 'predictions': []} for receptor in target_receptor_list}

    with torch.no_grad():
        for sequence, target in valid_loader:
            sequence = sequence.to(device)
            target   = target.to(device)

            outputs  = model(sequence)
            loss     = 0

            for i, receptor in enumerate(target_receptor_list):
                start_index      = energy_num * i
                end_index        = start_index + energy_num
                target_receptor  = target[:, start_index:end_index]

                receptor_mean    = outputs[i, :, 0] 
                receptor_var     = outputs[i, :, 1] 

                receptor_loss    = criterion(receptor_mean.unsqueeze(-1).expand_as(target_receptor), 
                                             target_receptor, 
                                             receptor_var.unsqueeze(-1).expand_as(target_receptor))
                loss            += receptor_loss

                results[receptor]['actuals'].append(target_receptor.cpu().numpy().tolist()) 
                results[receptor]['predictions'].append(receptor_mean.detach().cpu().numpy().tolist())
                

            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    return avg_loss, results


def plot_and_save(actuals_dict, predictions_dict, save_path):
    num_receptors = len(actuals_dict)
    fig, axs = plt.subplots(1, num_receptors, figsize=(6 * num_receptors, 6))
    

    if num_receptors == 1:
        axs = [axs]

    for ax, (receptor, actuals_list) in zip(axs, actuals_dict.items()):
        actuals_flat = np.array([item for sublist in actuals_list for item in sublist])
        predictions_flat = np.array([item for sublist in predictions_dict[receptor] for item in sublist])

        actuals_means = np.array_split(actuals_flat, len(predictions_flat))
        actuals_means = [np.mean(chunk) for chunk in actuals_means]

        if len(actuals_means) > 1 and len(predictions_flat) > 1:
            corr, _ = pearsonr(actuals_means, predictions_flat)
            correlation_text = f'Correlation: {corr:.2f}'
        else:
            correlation_text = 'Correlation: undefined'
        
        min_val = min(min(actuals_means), min(predictions_flat))
        max_val = max(max(actuals_means), max(predictions_flat))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

        ax.scatter(actuals_means, predictions_flat, alpha=0.5)
        ax.set_xlabel('Actual Means')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{receptor} Actual Means vs. Predictions\n{correlation_text}')
        ax.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()