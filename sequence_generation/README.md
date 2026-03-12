# sequence_generation

Tools for EvoDiff finetuning, sequence generation, and a CPP classifier.

## Environment

```
conda env create -f environment.yml
conda activate sequence_generation
```

## Data Layout

- `data/cellppd/` : CPP training sequences (`cpp_data_1.txt`, `cpp_data_2.txt`, `cpp_data_3.txt`)
- `models/evodiff/` : downloaded pretrained EvoDiff weights (`*.tar`)
- `models/paper/` : fine-tuned checkpoints for generation (optional)
- `data/sequences/` : generated sequence CSVs (output)
- `models/predictor/` : classifier outputs

## Finetuning

```
python scripts/finetune.py --model-type OADM
```

Key options (see `--help` for full list):
- `--epochs`
- `--generate-num`
- `--model-type` (`OADM` or `D3PM`)
- `--lora-*`

Pretrained weights are auto-downloaded to `models/evodiff/` if missing.

## Generation

```
python scripts/generate.py \
  --model-path models/paper/EvoDiff_Finetuning_640M_state_dict.pt \
  --model-type OADM \
  --model-scale 640M \
  --seq-len-min 9 --seq-len-max 18 \
  --generate-number 1000
```

Output goes to `data/sequences/`.

## Classifier

Train:
```
python scripts/classifier.py --mode train
```

Predict:
```
python scripts/classifier.py --mode predict
```

Outputs:
- `models/predictor/best_model_state_dict.pt`
- `models/predictor/lora_predictions.csv`
- `models/predictor/oadm_predictions.csv`
