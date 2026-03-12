#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from evodiff.collaters import D3PMCollater
from evodiff.generate import generate_oaardm, generate_d3pm
from evodiff.utils import Tokenizer

# Import model definition and LoRA setup from finetune_rewrite
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_rewrite import EvodiffConfig, LoRAConfig, build_model


# Architecture differences between model scales (all other params are shared)
_SCALE_OVERRIDES = {
    "38M":  dict(d_model=1024, n_layers=16, slim=True),
    "640M": dict(d_model=1280, n_layers=56, slim=False),
}

def make_evodiff_config(model_scale: str) -> EvodiffConfig:
    overrides = _SCALE_OVERRIDES.get(model_scale)
    if overrides is None:
        raise ValueError(f"Unknown model_scale: {model_scale!r}. Choose from {list(_SCALE_OVERRIDES)}")
    return EvodiffConfig(**overrides)


def load_finetuned_model(model_type: str, model_scale: str,
                         lora_weights_path: Path, downloads_dir: Path,
                         lora_cfg: LoRAConfig, device: torch.device):
    """Build model with pretrained weights, then overlay LoRA adapter weights."""
    evodiff_cfg = make_evodiff_config(model_scale)
    model = build_model(evodiff_cfg, lora_cfg, model_type, downloads_dir)
    lora_state = torch.load(str(lora_weights_path), map_location="cpu")
    model.load_state_dict(lora_state, strict=False)
    model.to(device)
    model.eval()
    return model, evodiff_cfg



def make_generation_components(model_type: str, diffusion_timesteps: int):
    """Return (tokenizer, Q, Q_bar, timesteps) for the given model type."""
    if model_type == "OADM":
        tokenizer = Tokenizer()
        return tokenizer, None, None, None
    tokenizer = Tokenizer(sequences=True)
    Q_bar, Q = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
    return tokenizer, Q, Q_bar, diffusion_timesteps



def generate_unique(model, tokenizer, model_type, seq_len, num_seqs, device,
                    Q=None, Q_bar=None, timesteps=None) -> list[str]:
    """Generate exactly `num_seqs` unique sequences of length `seq_len`."""
    unique: set[str] = set()
    while len(unique) < num_seqs:
        batch_size = min(num_seqs - len(unique), 10000)
        if model_type == "OADM":
            _, seqs = generate_oaardm(model, tokenizer, seq_len,
                                      batch_size=batch_size, device=device)
        else:
            _, seqs = generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len,
                                    batch_size=batch_size, device=device)
        unique.update(seqs)
    return list(unique)[:num_seqs]



def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parents[1]
    downloads_dir = repo_root / "models" / "evodiff"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Generate sequences from a fine-tuned EvoDiff model.")

    # Model
    parser.add_argument("--model-path",  required=True, type=Path,
                        help="Path to LoRA state_dict (.pt) saved by finetune_rewrite.py.")
    parser.add_argument("--model-type",  choices=["OADM", "D3PM"], default="OADM")
    parser.add_argument("--model-scale", choices=["38M", "640M"],  default="640M")

    # Generation
    parser.add_argument("--seq-len-min",     type=int, default=9)
    parser.add_argument("--seq-len-max",     type=int, default=18)
    parser.add_argument("--generate-number", type=int, default=1000,
                        help="Number of unique sequences to generate per length.")
    parser.add_argument("--output-dir", type=Path,
                        default=repo_root / "data" / "generated_data")

    def str2bool(v):
        return v.lower() in ("1", "true", "yes", "y") if isinstance(v, str) else bool(v)

    parser.add_argument("--use-lora-upembedder",  type=str2bool, default=False)
    parser.add_argument("--use-lora-bn-seq1",     type=str2bool, default=False)
    parser.add_argument("--use-lora-bn-seq2",     type=str2bool, default=False)
    parser.add_argument("--use-lora-decoder",     type=str2bool, default=False)
    parser.add_argument("--use-lora-maskedconv1", type=str2bool, default=False,
                        help="bmm-based LoRA (ConvLoRA_v1) on MaskedConv1d layers.")
    parser.add_argument("--use-lora-maskedconv2", type=str2bool, default=True,
                        help="matmul-based LoRA (ConvLoRA_v2) on MaskedConv1d layers.")
    parser.add_argument("--lora-r",       type=int,   default=8)
    parser.add_argument("--lora-alpha",   type=int,   default=2)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    args = parser.parse_args()

    if args.use_lora_maskedconv1 and args.use_lora_maskedconv2:
        parser.error("--use-lora-maskedconv1 and --use-lora-maskedconv2 cannot both be set.")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, evodiff_cfg = load_finetuned_model(
        model_type       = args.model_type,
        model_scale      = args.model_scale,
        lora_weights_path= args.model_path,
        downloads_dir    = downloads_dir,
        lora_cfg         = lora_cfg,
        device           = device,
    )

    tokenizer, Q, Q_bar, timesteps = make_generation_components(
        args.model_type, evodiff_cfg.diffusion_timesteps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_dt = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"{current_dt}_{args.seq_len_min}-{args.seq_len_max}length.csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['Sequence'])

    for seq_len in range(args.seq_len_min, args.seq_len_max + 1):
        sequences = generate_unique(
            model=model, tokenizer=tokenizer, model_type=args.model_type,
            seq_len=seq_len, num_seqs=args.generate_number, device=device,
            Q=Q, Q_bar=Q_bar, timesteps=timesteps,
        )
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for seq in sequences:
                writer.writerow([seq])
        print(f"  seq_len={seq_len}: {len(sequences)} sequences generated")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
