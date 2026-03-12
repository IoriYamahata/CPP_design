
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


AMINO_ACID_DICT = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,  "G": 6,  "H": 7,  "I": 8,
    "K": 9,  "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "PAD": 0,
}


def encode_sequences(seqs: pd.Series, max_len: int) -> torch.Tensor:
    out = []
    for s in seqs:
        if pd.isna(s):
            s = ""
        s = str(s).strip()
        if len(s) > max_len:
            s = s[:max_len]

        encoded = [AMINO_ACID_DICT.get(aa, 0) for aa in s]
        padded = encoded + [AMINO_ACID_DICT["PAD"]] * (max_len - len(encoded))
        out.append(padded)

    return torch.tensor(out, dtype=torch.long)


class CPPDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


class CPPClassifier(nn.Module):
    def __init__(self, num_amino_acids: int, d_model: int, hidden_size: int, dropout_prob: float, max_seq_length: int):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.embedding  = nn.Embedding(num_amino_acids, d_model, padding_idx=AMINO_ACID_DICT["PAD"])

        self.conv1      = nn.Conv1d(in_channels=d_model, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1        = nn.BatchNorm1d(hidden_size)
        self.dropout1   = nn.Dropout(dropout_prob)

        self.conv2      = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2)
        self.bn2        = nn.BatchNorm1d(hidden_size)
        self.dropout2   = nn.Dropout(dropout_prob)

        self.fc         = nn.Linear(hidden_size * max_seq_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) 
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = x.permute(0, 2, 1)
        x = torch.flatten(x, start_dim=1)

        logits = self.fc(x)
        return logits


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for (batch_data,) in dataloader:
        batch_data = batch_data.to(device)
        logits = model(batch_data)
        prob = torch.sigmoid(logits).cpu().numpy()  # (B, 1)
        preds.append(prob)
    return np.concatenate(preds, axis=0).reshape(-1)


def _infer_label_column(df: pd.DataFrame) -> str:
    candidates = ["class", "Class", "label", "Label", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Label column not found. Existing columns: {list(df.columns)}")


def _infer_sequence_column(df: pd.DataFrame) -> str:
    candidates = ["sequence", "Sequence", "seq", "Seq"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Sequence column not found. Existing columns: {list(df.columns)}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    models_dir = repo_root / "models" / "predictor"
    models_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train CPP classifier and run predictions.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")

    parser.add_argument("--filtered-csv", default=str(data_dir / "processed_data" / "filtered_sequence.csv"))
    parser.add_argument("--test-csv", default=str(data_dir / "processed_data" / "biomolecules-2162660_Supplementary Spreadsheets S1 Test Set.csv"))
    parser.add_argument("--lora-csv", default=str(data_dir / "generated_data" / "LoRA_9-18_residue.csv"))
    parser.add_argument("--default-csv", default=str(data_dir / "generated_data" / "default_9-18_residue.csv"))

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--split", type=float, default=0.8)

    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=16)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = models_dir / "best_checkpoint.pt"

    if args.mode == "train":
        filtered_df = pd.read_csv(args.filtered_csv)

        seq_col = _infer_sequence_column(filtered_df)
        lab_col = _infer_label_column(filtered_df)

        max_len = filtered_df[seq_col].astype(str).apply(len).max()
        sequences = encode_sequences(filtered_df[seq_col], max_len)

        raw = filtered_df[lab_col]
        if raw.dtype == object:
            labels_np = raw.apply(lambda x: 1 if str(x).strip().upper() == "CPP" else 0).astype(np.float32).values
        else:
            labels_np = pd.to_numeric(raw, errors="raise").astype(np.float32).values
        labels = torch.tensor(labels_np, dtype=torch.float32).unsqueeze(1)

        train_data, val_data, train_labels, val_labels = train_test_split(
            sequences, labels, test_size=1 - args.split, random_state=42, stratify=labels_np
        )

        train_loader = DataLoader(CPPDataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(CPPDataset(val_data, val_labels), batch_size=args.batch_size, shuffle=False)

        model = CPPClassifier(
            num_amino_acids=len(AMINO_ACID_DICT),
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            dropout_prob=args.dropout,
            max_seq_length=max_len,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for batch_data, batch_label in train_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_data)
                loss = criterion(logits, batch_label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_label in val_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    logits = model(batch_data)
                    loss = criterion(logits, batch_label)
                    val_loss += loss.item()

            avg_train = running_loss / max(1, len(train_loader))
            avg_val = val_loss / max(1, len(val_loader))

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "max_len": max_len,
                        "num_amino_acids": len(AMINO_ACID_DICT),
                        "d_model": args.d_model,
                        "hidden_size": args.hidden_size,
                        "dropout": args.dropout,
                    },
                    best_path,
                )

            print(f"Epoch {epoch+1}/{args.epochs} - train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        print(f"Saved best checkpoint: {best_path}")

        # --- Independent test set evaluation ---
        test_df = pd.read_csv(args.test_csv)
        test_seq_col = _infer_sequence_column(test_df)
        test_lab_col = _infer_label_column(test_df)

        test_seqs = encode_sequences(test_df[test_seq_col], max_len)
        test_raw = test_df[test_lab_col]
        if test_raw.dtype == object:
            test_labels_np = test_raw.apply(lambda x: 1 if str(x).strip().upper() == "CPP" else 0).astype(np.float32).values
        else:
            test_labels_np = pd.to_numeric(test_raw, errors="raise").astype(np.float32).values
        test_labels = torch.tensor(test_labels_np, dtype=torch.float32).unsqueeze(1)

        test_loader = DataLoader(CPPDataset(test_seqs, test_labels), batch_size=args.batch_size, shuffle=False)

        # Reload best checkpoint for evaluation
        best_ckpt = torch.load(best_path, map_location=device)
        best_model = CPPClassifier(
            num_amino_acids=int(best_ckpt["num_amino_acids"]),
            d_model=int(best_ckpt["d_model"]),
            hidden_size=int(best_ckpt["hidden_size"]),
            dropout_prob=float(best_ckpt["dropout"]),
            max_seq_length=int(best_ckpt["max_len"]),
        ).to(device)
        best_model.load_state_dict(best_ckpt["state_dict"])
        best_model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_label in test_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                logits = best_model(batch_data)
                test_loss += criterion(logits, batch_label).item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == batch_label).sum().item()
                total += batch_label.size(0)

        avg_test_loss = test_loss / max(1, len(test_loader))
        accuracy = correct / total if total > 0 else 0.0
        print(f"Test set results: loss={avg_test_loss:.4f}, accuracy={accuracy:.4f} ({correct}/{total})")

    if not best_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    max_len = int(ckpt["max_len"])

    model = CPPClassifier(
        num_amino_acids=int(ckpt["num_amino_acids"]),
        d_model=int(ckpt["d_model"]),
        hidden_size=int(ckpt["hidden_size"]),
        dropout_prob=float(ckpt["dropout"]),
        max_seq_length=max_len,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    for label, path in [("lora", args.lora_csv), ("default", args.default_csv)]:
        p = Path(path)
        if not p.exists():
            print(f"Skip prediction: {p} not found")
            continue

        df = pd.read_csv(p)
        seq_col = _infer_sequence_column(df)

        seqs = encode_sequences(df[seq_col], max_len)
        dl = DataLoader(TensorDataset(seqs), batch_size=args.batch_size, shuffle=False)

        preds = predict(model, dl, device)

        out_path = models_dir / f"{label}_predictions.csv"
        pd.DataFrame({seq_col: df[seq_col], "Prediction": preds}).to_csv(out_path, index=False)
        print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()