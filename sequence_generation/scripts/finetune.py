#!/usr/bin/env python
from __future__ import annotations

import os
import re
import csv
import sys
import math

import argparse
import torch
import numpy as np
import pandas as pd
import loralib as lora
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from datetime import datetime, timezone

from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from loralib.layers import Conv1d, LoRALayer
from sequence_models.constants import MSA_ALPHABET
from sequence_models.metrics import MaskedAccuracy
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

from evodiff.utils import Tokenizer, download_model
from evodiff.collaters import OAMaskCollater, D3PMCollater
from evodiff.generate import generate_oaardm, generate_d3pm
from evodiff.losses import OAMaskedCrossEntropyLoss, D3PMCELoss, D3PMLVBLoss


@dataclass
class EvodiffConfig:
    """Architecture hyperparameters — must match pretrained weights."""
    n_tokens:           int   = len(MSA_ALPHABET)
    d_embed:            int   = 8
    d_model:            int   = 1280
    n_layers:           int   = 56
    kernel_size:        int   = 5
    r:                  int   = 128     # dilation cycle length
    activation:         str   = "gelu"
    slim:               bool  = False
    dropout:            float = 0.0
    padding_idx:        int   = 28
    tie_weights:        bool  = False
    final_norm:         bool  = True
    diffusion_timesteps: int  = 500


@dataclass
class LoRAConfig:
    """LoRA adapter config: target layers and adapter hyperparams."""
    # Which layers receive LoRA adapters
    use_upembedder:   bool  = False
    use_bn_seq1:      bool  = False   # sequence1 PositionFeedForward
    use_bn_seq2:      bool  = False   # sequence2 PositionFeedForward
    use_decoder:      bool  = False
    use_maskedconv1:  bool  = False   # bmm-based LoRA on MaskedConv1d
    use_maskedconv2:  bool  = True    # matmul-based LoRA on MaskedConv1d
    # Adapter hyperparams
    r:        int   = 8
    alpha:    int   = 2
    dropout:  float = 0.0


@dataclass
class TrainingConfig:
    """Training / optimisation hyperparameters."""
    epochs:        int   = 3000
    batch_size:    int   = 100
    initial_lr:    float = 1e-7
    max_lr:        float = 1e-4
    final_lr:      float = 1e-7
    weight_decay:  float = 0.0
    # LR schedule shape: linear ramp → flat hold → exponential decay
    ramp_up_frac:  float = 0.1   # fraction of epochs for warm-up
    hold_frac:     float = 0.5   # fraction of epochs to hold max_lr
    # D3PM-specific
    reweighting_term: float = 0.0  # λ: loss = (lvb + λ·ce) × n_tokens


class PositionFeedForward(nn.Module):
    """Standard 1×1 conv (no LoRA)."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class LoRAPositionFeedForward(nn.Module):
    """1×1 conv with LoRA adapter."""
    def __init__(self, d_in: int, d_out: int, lora_cfg: LoRAConfig):
        super().__init__()
        self.conv = Conv1d(d_in, d_out, kernel_size=1,
                           r=lora_cfg.r, lora_alpha=lora_cfg.alpha,
                           lora_dropout=lora_cfg.dropout)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv1d(nn.Conv1d):
    """Conv1d with symmetric padding and optional input masking."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, dilation=dilation, groups=groups,
                         bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvLoRA_v1(nn.Module, LoRALayer):
    """LoRA for Conv1d via batch-matrix-multiply.

    lora_A: (kernel_size, in_channels, r)
    lora_B: (kernel_size, r, out_channels)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 lora_cfg: LoRAConfig, merge_weights=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=lora_cfg.r, lora_alpha=lora_cfg.alpha,
                           lora_dropout=lora_cfg.dropout, merge_weights=merge_weights)
        if lora_cfg.r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((kernel_size, in_channels, lora_cfg.r)))
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((kernel_size, lora_cfg.r, out_channels // self.conv.groups)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _lora_weight(self):
        return torch.permute(torch.bmm(self.lora_A, self.lora_B), (2, 1, 0))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.merge_weights and self.merged and self.r > 0:
            self.conv.weight.data -= self._lora_weight().view(self.conv.weight.shape) * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged and self.r > 0:
            self.conv.weight.data += self._lora_weight().view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            weight = self.conv.weight + self._lora_weight() * self.scaling
            return F.conv1d(x, weight, self.conv.bias, self.conv.stride,
                            self.conv.padding, self.conv.dilation, self.conv.groups)
        return self.conv(x)


class ConvLoRA_v2(nn.Module, LoRALayer):
    """LoRA for Conv1d via matrix multiply.

    lora_A: (r, in_channels * kernel_size)
    lora_B: (out_channels, r)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 lora_cfg: LoRAConfig, merge_weights=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=lora_cfg.r, lora_alpha=lora_cfg.alpha,
                           lora_dropout=lora_cfg.dropout, merge_weights=merge_weights)
        if lora_cfg.r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((lora_cfg.r, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups, lora_cfg.r)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.merge_weights and self.merged and self.r > 0:
            self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged and self.r > 0:
            self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias,
            )
        return self.conv(x)


class LoRAMaskedConv1d_v1(ConvLoRA_v1):
    """MaskedConv1d with bmm-based LoRA (ConvLoRA_v1)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True,
                 lora_cfg: LoRAConfig = None, merge_weights=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, lora_cfg=lora_cfg,
                         stride=stride, dilation=dilation, groups=groups,
                         bias=bias, padding=padding, merge_weights=merge_weights)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class LoRAMaskedConv1d_v2(ConvLoRA_v2):
    """MaskedConv1d with matmul-based LoRA (ConvLoRA_v2)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True,
                 lora_cfg: LoRAConfig = None, merge_weights=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, lora_cfg=lora_cfg,
                         stride=stride, dilation=dilation, groups=groups,
                         bias=bias, padding=padding, merge_weights=merge_weights)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1,
                 activation='relu', lora_cfg: LoRAConfig = None):
        super().__init__()

        # Conv layer (plain or LoRA-wrapped)
        if lora_cfg is not None and lora_cfg.use_maskedconv1:
            self.conv = LoRAMaskedConv1d_v1(d_h, d_h, kernel_size=kernel_size,
                                             dilation=dilation, groups=groups, lora_cfg=lora_cfg)
        elif lora_cfg is not None and lora_cfg.use_maskedconv2:
            self.conv = LoRAMaskedConv1d_v2(d_h, d_h, kernel_size=kernel_size,
                                             dilation=dilation, groups=groups, lora_cfg=lora_cfg)
        else:
            self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size,
                                     dilation=dilation, groups=groups)

        act = nn.ReLU if activation == 'relu' else nn.GELU

        # FFN layers (plain or LoRA-wrapped)
        pff1 = (LoRAPositionFeedForward(d_in, d_h, lora_cfg)
                if lora_cfg is not None and lora_cfg.use_bn_seq1
                else PositionFeedForward(d_in, d_h))
        pff2 = (LoRAPositionFeedForward(d_h, d_out, lora_cfg)
                if lora_cfg is not None and lora_cfg.use_bn_seq2
                else PositionFeedForward(d_h, d_out))

        self.sequence1 = nn.Sequential(nn.LayerNorm(d_in), act(), pff1, nn.LayerNorm(d_h), act())
        self.sequence2 = nn.Sequential(nn.LayerNorm(d_h), act(), pff2)

    def forward(self, x, input_mask=None):
        return x + self.sequence2(self.conv(self.sequence1(x), input_mask=input_mask))


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""
    def __init__(self, d_model: int, length: int):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {self.d_model}")
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.to(x.device)[x]


class ByteNetTime(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, dropout=0.0, slim=True, activation='relu',
                 timesteps=None, lora_cfg: LoRAConfig = None):
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps)
        self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)

        if lora_cfg is not None and lora_cfg.use_upembedder:
            self.up_embedder = LoRAPositionFeedForward(d_embedding, d_model, lora_cfg)
        else:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)

        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model // 2 if slim else d_model
        self.layers = nn.ModuleList([
            ByteNetBlock(d_model, d_h, d_model, kernel_size,
                         dilation=d, activation=activation, lora_cfg=lora_cfg)
            for d in dilations
        ])
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x)                                  # (B, L, d_embed)
        e2 = self.time_encoding(y)                            # (B, d_embed)
        e2 = e2.expand(e.shape[1], e2.shape[0], e2.shape[1]) # (L, B, d_embed)
        e2 = e2.reshape(e.shape[0], e.shape[1], e.shape[2])  # (B, L, d_embed)
        e = e + e2
        e = self.up_embedder(e)
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, dropout=0.0, final_ln=False, slim=True,
                 activation='relu', tie_weights=False, timesteps=None,
                 lora_cfg: LoRAConfig = None):
        super().__init__()
        self.embedder = ByteNetTime(
            n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
            padding_idx=padding_idx, dropout=dropout, slim=slim,
            activation=activation, timesteps=timesteps, lora_cfg=lora_cfg)

        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        elif lora_cfg is not None and lora_cfg.use_decoder:
            self.decoder = LoRAPositionFeedForward(d_model, n_tokens, lora_cfg)
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)

        self.last_norm = nn.LayerNorm(d_model) if final_ln else nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)



def _remap_state_dict_keys(state_dict: dict, lora_cfg: LoRAConfig) -> dict:
    """Rename pretrained weight keys to match LoRA-wrapped layer names."""
    if lora_cfg.use_upembedder:
        state_dict = {
            k.replace("embedder.up_embedder.conv.", "embedder.up_embedder.conv.conv."): v
            for k, v in state_dict.items()}
    if lora_cfg.use_bn_seq1:
        state_dict = {
            k.replace("sequence1.2.conv.", "sequence1.2.conv.conv."): v
            for k, v in state_dict.items()}
    if lora_cfg.use_bn_seq2:
        state_dict = {
            k.replace("sequence2.2.conv.", "sequence2.2.conv.conv."): v
            for k, v in state_dict.items()}
    if lora_cfg.use_decoder:
        state_dict = {
            k.replace("decoder.conv.", "decoder.conv.conv."): v
            for k, v in state_dict.items()}
    if lora_cfg.use_maskedconv1 or lora_cfg.use_maskedconv2:
        state_dict = {_replace_conv_key(k): v for k, v in state_dict.items()}
    return state_dict


def _replace_conv_key(key: str) -> str:
    """Rewrite layers.N.conv.{weight,bias} → layers.N.conv.conv.{weight,bias}."""
    for pattern, replacement in [
        (r'(layers\.\d+\.conv)(\.weight)', r'\1.conv\2'),
        (r'(layers\.\d+\.conv)(\.bias)',   r'\1.conv\2'),
    ]:
        new_key = re.sub(pattern, replacement, key)
        if new_key != key:
            return new_key
    return key


def build_model(evodiff_cfg: EvodiffConfig, lora_cfg: LoRAConfig,
                model_type: str, downloads_dir: Path) -> ByteNetLMTime:
    """Instantiate ByteNetLMTime, load pretrained weights, and freeze non-LoRA params."""
    model = ByteNetLMTime(
        n_tokens    = evodiff_cfg.n_tokens,
        d_embedding = evodiff_cfg.d_embed,
        d_model     = evodiff_cfg.d_model,
        n_layers    = evodiff_cfg.n_layers,
        kernel_size = evodiff_cfg.kernel_size,
        r           = evodiff_cfg.r,
        padding_idx = evodiff_cfg.padding_idx,
        dropout     = evodiff_cfg.dropout,
        tie_weights = evodiff_cfg.tie_weights,
        final_ln    = evodiff_cfg.final_norm,
        slim        = evodiff_cfg.slim,
        activation  = evodiff_cfg.activation,
        timesteps   = evodiff_cfg.diffusion_timesteps,
        lora_cfg    = lora_cfg,
    )

    model_name = "oaar-640M" if model_type == "OADM" else "d3pm-uniform-640M"
    local_path = downloads_dir / f"{model_name}.tar"
    if not local_path.exists():
        print(f"Downloading {model_name} ...")
        torch.save(download_model(model_name), str(local_path))

    raw = torch.load(str(local_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in raw["model_state_dict"].items()}
    state_dict = _remap_state_dict_keys(state_dict, lora_cfg)
    model.load_state_dict(state_dict, strict=False)

    lora.mark_only_lora_as_trainable(model)
    return model



def load_cpp_dataframe(data_dir: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(data_dir / f"cpp_data_{i}.txt", header=None) for i in range(1, 4)]
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df = pd.DataFrame(df[0].str.upper())
    df.columns = ['sequence']
    return df


class CPPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = 2048):
        self.sequences = df['sequence'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            start = np.random.randint(len(seq) - self.max_len)
            seq = seq[start: start + self.max_len]
        return (seq,)



def _lr_schedule(epoch, initial_lr, max_lr, ramp_up_epochs, hold_epochs, gamma):
    if epoch < ramp_up_epochs:
        return (max_lr / initial_lr) ** (epoch / ramp_up_epochs)
    if epoch < hold_epochs:
        return max_lr / initial_lr
    return max_lr / initial_lr * (gamma ** (epoch - hold_epochs))


def epoch_oadm(model, dataloader, optimizer, scheduler, epoch_num,
               train=True, scaler=None, device=None, padding_idx=None,
               loss_func=None, accu_func=None):
    model.train() if train else model.eval()
    total_loss = total_nll = total_accu = total_tokens = 0.0

    for idx, batch in enumerate(dataloader, 1):
        src, timestep, tgt, mask = [t.to(device) for t in batch]
        input_mask = (src != padding_idx).float()
        n_tokens = input_mask.sum()

        if train:
            optimizer.zero_grad()
        with autocast(dtype=torch.float32):
            outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))
            ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)
            accu = accu_func(outputs, tgt, mask) * n_tokens
        if train:
            scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss   += ce_loss.item()
        total_nll    += nll_loss.item()
        total_accu   += accu.item()
        total_tokens += n_tokens.item()

    if train:
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_num} Train - Loss: {total_loss/idx:.4f}, "
              f"NLL: {total_nll/idx:.4f}, Acc: {total_accu/total_tokens:.4f}, LR: {lr:.4e}")
    else:
        print(f"Epoch {epoch_num} Valid - Loss: {total_loss/idx:.4f}, "
              f"NLL: {total_nll/idx:.4f}, Acc: {total_accu/total_tokens:.4f}")

    return total_loss / idx, total_nll / idx, total_accu / total_tokens


def epoch_d3pm(model, dataloader, optimizer, scheduler, epoch_num,
               train=True, scaler=None, device=None, padding_idx=None,
               loss_func1=None, loss_func2=None, reweighting_term=0.0, accu_func=None):
    model.train() if train else model.eval()
    total_loss = total_nll = total_accu = total_tokens = 0.0

    for idx, batch in enumerate(dataloader, 1):
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
        src, tgt, timestep = src.to(device), tgt.to(device), timestep.to(device)
        src_onehot, tgt_onehot = src_onehot.to(device), tgt_onehot.to(device)
        Q, Q_bar, q = Q.to(device), Q_bar.to(device), q.to(device)
        input_mask = (src != padding_idx).float()
        n_tokens = input_mask.sum()

        if train:
            optimizer.zero_grad()
        with autocast(dtype=torch.float32):
            outputs  = model(src, timestep, input_mask=input_mask.unsqueeze(-1))
            lvb_loss = loss_func1(src_onehot, q, outputs, tgt, tgt_onehot,
                                  input_mask, timestep, Q, Q_bar).to(torch.float32)
            ce_loss  = loss_func2(outputs, tgt, input_mask).to(torch.float32)
            loss     = (lvb_loss + reweighting_term * ce_loss) * n_tokens
            nll_loss = ce_loss * n_tokens
            accu     = accu_func(outputs, tgt, input_mask) * n_tokens
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss   += loss.item()
        total_nll    += nll_loss.item()
        total_accu   += accu.item()
        total_tokens += n_tokens.item()

    if train:
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_num} Train - Loss: {total_loss/idx:.4f}, "
              f"NLL: {total_nll/idx:.4f}, Acc: {total_accu/total_tokens:.4f}, LR: {lr:.4e}")
    else:
        print(f"Epoch {epoch_num} Valid - Loss: {total_loss/idx:.4f}, "
              f"NLL: {total_nll/idx:.4f}, Acc: {total_accu/total_tokens:.4f}")

    return total_loss / idx, total_nll / idx, total_accu / total_tokens


def finetune(model, dl_train, dl_valid, epoch_fn, optimizer, scheduler,
             scaler, epochs: int):
    train_losses, valid_losses = [], []
    for e in range(1, epochs + 1):
        train_loss, _, _ = epoch_fn(model, dl_train, optimizer, scheduler, e,
                                    train=True, scaler=scaler)
        valid_loss, _, _ = epoch_fn(model, dl_valid, None, None, e,
                                    train=False, scaler=None)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    return train_losses, valid_losses



def print_trainable_parameters(model: nn.Module) -> None:
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        (trainable if param.requires_grad else frozen).append((name, param.numel()))
    print("Trainable parameters:")
    for name, count in trainable:
        print(f"  {name}: {count:,}")
    print(f"\n  Total trainable: {sum(c for _, c in trainable):,}")
    print(f"  Total frozen:    {sum(c for _, c in frozen):,}")


def generate_and_save(model, tokenizer, model_type, seq_len, num_seqs,
                      output_csv, device, Q=None, Q_bar=None, timesteps=None):
    model.eval()
    if model_type == "OADM":
        _, sequences = generate_oaardm(model, tokenizer, seq_len,
                                        batch_size=num_seqs, device=device)
    else:
        _, sequences = generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len,
                                      batch_size=num_seqs, device=device)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Sequence'])
        for seq in sequences:
            writer.writerow([seq])


def save_run_log(qc_path, graph_path, start_time, model_type,
                 lora_cfg: LoRAConfig, train_cfg: TrainingConfig,
                 train_losses, valid_losses, state_dict_keys):
    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, valid_losses, label='Valid loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    # QC text
    end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(qc_path, 'w') as f:
        f.write(f"File:  {os.path.basename(qc_path)}\n")
        f.write(f"Start: {start_time}\n")
        f.write(f"End:   {end_time}\n\n")
        f.write(f"Algorithm: {model_type}\n")
        if model_type == "D3PM":
            f.write(f"  loss = (lvb + {train_cfg.reweighting_term} * ce) * n_tokens\n")
        f.write(f"\n[LoRA]\n")
        f.write(f"  r:              {lora_cfg.r}\n")
        f.write(f"  alpha:          {lora_cfg.alpha}\n")
        f.write(f"  dropout:        {lora_cfg.dropout}\n")
        f.write(f"  use_upembedder: {lora_cfg.use_upembedder}\n")
        f.write(f"  use_bn_seq1:    {lora_cfg.use_bn_seq1}\n")
        f.write(f"  use_bn_seq2:    {lora_cfg.use_bn_seq2}\n")
        f.write(f"  use_decoder:    {lora_cfg.use_decoder}\n")
        f.write(f"  use_maskedconv1:{lora_cfg.use_maskedconv1}\n")
        f.write(f"  use_maskedconv2:{lora_cfg.use_maskedconv2}\n")
        f.write(f"\n[Training]\n")
        f.write(f"  epochs:           {train_cfg.epochs}\n")
        f.write(f"  batch_size:       {train_cfg.batch_size}\n")
        f.write(f"  initial_lr:       {train_cfg.initial_lr}\n")
        f.write(f"  max_lr:           {train_cfg.max_lr}\n")
        f.write(f"  final_lr:         {train_cfg.final_lr}\n")
        f.write(f"  weight_decay:     {train_cfg.weight_decay}\n")
        f.write(f"\nTrain losses: {train_losses}\n")
        f.write(f"Valid losses: {valid_losses}\n")
        f.write("\nState dict keys:\n")
        for k in state_dict_keys:
            f.write(f"  {k}\n")



def main() -> None:
    repo_root     = Path(__file__).resolve().parents[1]
    downloads_dir = repo_root / "models" / "evodiff"
    results_dir   = repo_root / "results"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    def str2bool(v):
        return v.lower() in ("1", "true", "yes", "y") if isinstance(v, str) else bool(v)

    parser = argparse.ArgumentParser(description="Fine-tune EvoDiff (640M) with LoRA on CPP data.")

    # Experiment
    parser.add_argument("--tag",              default="EvoDiff_Finetuning_640M")
    parser.add_argument("--experiment-index", type=int,  default=1)
    parser.add_argument("--data-dir",         type=Path, default=repo_root / "data" / "cellppd")
    parser.add_argument("--model-type",       choices=["OADM", "D3PM"], default="OADM")
    parser.add_argument("--seq-len",          type=int,  default=20)
    parser.add_argument("--generate-num",     type=int,  default=10000)

    # LoRA
    parser.add_argument("--use-lora-upembedder",  type=str2bool, default=False)
    parser.add_argument("--use-lora-bn-seq1",     type=str2bool, default=False)
    parser.add_argument("--use-lora-bn-seq2",     type=str2bool, default=False)
    parser.add_argument("--use-lora-decoder",     type=str2bool, default=False)
    parser.add_argument("--use-lora-maskedconv1", type=str2bool, default=False,
                        help="bmm-based LoRA (ConvLoRA_v1) on MaskedConv1d layers")
    parser.add_argument("--use-lora-maskedconv2", type=str2bool, default=True,
                        help="matmul-based LoRA (ConvLoRA_v2) on MaskedConv1d layers")
    parser.add_argument("--lora-r",       type=int,   default=8)
    parser.add_argument("--lora-alpha",   type=int,   default=2)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs",           type=int,   default=3000)
    parser.add_argument("--batch-size",       type=int,   default=100)
    parser.add_argument("--initial-lr",       type=float, default=1e-7)
    parser.add_argument("--max-lr",           type=float, default=1e-4)
    parser.add_argument("--final-lr",         type=float, default=1e-7)
    parser.add_argument("--weight-decay",     type=float, default=0.0)
    parser.add_argument("--reweighting-term", type=float, default=0.0,
                        help="D3PM λ: loss = (lvb + λ·ce) * n_tokens")

    args = parser.parse_args()

    evodiff_cfg = EvodiffConfig()
    lora_cfg = LoRAConfig(
        use_upembedder  = args.use_lora_upembedder,
        use_bn_seq1     = args.use_lora_bn_seq1,
        use_bn_seq2     = args.use_lora_bn_seq2,
        use_decoder     = args.use_lora_decoder,
        use_maskedconv1 = args.use_lora_maskedconv1,
        use_maskedconv2 = args.use_lora_maskedconv2,
        r               = args.lora_r,
        alpha           = args.lora_alpha,
        dropout         = args.lora_dropout,
    )
    train_cfg = TrainingConfig(
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        initial_lr       = args.initial_lr,
        max_lr           = args.max_lr,
        final_lr         = args.final_lr,
        weight_decay     = args.weight_decay,
        reweighting_term = args.reweighting_term,
    )

    start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(evodiff_cfg, lora_cfg, args.model_type, downloads_dir)
    print_trainable_parameters(model)
    model.to(device)

    if args.model_type == "OADM":
        tokenizer = Tokenizer()
        collater  = OAMaskCollater(tokenizer=tokenizer)
        Q = Q_bar = timesteps = None
    else:
        timesteps = evodiff_cfg.diffusion_timesteps
        tokenizer = Tokenizer(sequences=True)
        Q_bar, Q  = tokenizer.q_random_schedule(timesteps=timesteps)
        collater  = D3PMCollater(tokenizer=tokenizer, num_timesteps=timesteps, Q=Q, Q_bar=Q_bar)

    padding_idx = tokenizer.pad_id
    print(f"Padding idx: {padding_idx}  Mask idx: {tokenizer.mask_id}")

    cpp_df = load_cpp_dataframe(args.data_dir)
    train_df, valid_df = train_test_split(cpp_df, test_size=0.1, random_state=42)
    dl_train = DataLoader(CPPDataset(train_df), shuffle=True,  batch_size=train_cfg.batch_size,
                          num_workers=4, collate_fn=collater)
    dl_valid = DataLoader(CPPDataset(valid_df), shuffle=False, batch_size=train_cfg.batch_size,
                          num_workers=4, collate_fn=collater)

    ramp_up_epochs = train_cfg.epochs * train_cfg.ramp_up_frac
    hold_epochs    = train_cfg.epochs * train_cfg.hold_frac
    decay_epochs   = train_cfg.epochs - hold_epochs
    gamma = (train_cfg.final_lr / train_cfg.max_lr) ** (1 / decay_epochs)

    optimizer = optim.Adam(model.parameters(),
                           lr=train_cfg.initial_lr, weight_decay=train_cfg.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: _lr_schedule(
        e, train_cfg.initial_lr, train_cfg.max_lr, ramp_up_epochs, hold_epochs, gamma))
    scaler = GradScaler()

    accu_func = MaskedAccuracy()
    if args.model_type == "OADM":
        loss_func = OAMaskedCrossEntropyLoss(reweight=True)
        epoch_fn = partial(epoch_oadm, device=device, padding_idx=padding_idx,
                           loss_func=loss_func, accu_func=accu_func)
    else:
        loss_func1 = D3PMLVBLoss(tmax=evodiff_cfg.diffusion_timesteps, tokenizer=tokenizer)
        loss_func2 = D3PMCELoss(tokenizer=tokenizer)
        epoch_fn = partial(epoch_d3pm, device=device, padding_idx=padding_idx,
                           loss_func1=loss_func1, loss_func2=loss_func2,
                           reweighting_term=train_cfg.reweighting_term, accu_func=accu_func)

    train_losses, valid_losses = finetune(
        model, dl_train, dl_valid, epoch_fn, optimizer, scheduler, scaler, train_cfg.epochs)

    current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_name = f"{current_date}_{args.tag}_run{args.experiment_index}"
    out_dir  = results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(lora.lora_state_dict(model), out_dir / f"{run_name}_state_dict.pt")

    generate_and_save(
        model=model, tokenizer=tokenizer, model_type=args.model_type,
        seq_len=args.seq_len, num_seqs=args.generate_num,
        output_csv=out_dir / f"{run_name}_generation.csv",
        device=device, Q=Q, Q_bar=Q_bar, timesteps=timesteps,
    )

    save_run_log(
        qc_path=out_dir / f"{run_name}_QC.txt",
        graph_path=out_dir / f"{run_name}_QC_graph.png",
        start_time=start_time,
        model_type=args.model_type,
        lora_cfg=lora_cfg,
        train_cfg=train_cfg,
        train_losses=train_losses,
        valid_losses=valid_losses,
        state_dict_keys=list(model.state_dict().keys()),
    )


if __name__ == "__main__":
    main()
