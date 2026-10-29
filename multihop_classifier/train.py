import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from feature_utils import extract_features, feature_vector
from model import BertWithFeatures

BASE_MODEL = "hfl/chinese-roberta-wwm-ext"


class QADataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["question"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        features = row.get("features") or extract_features(row["question"])
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "features": torch.tensor(feature_vector(features), dtype=torch.float32),
            "label": torch.tensor(1 if row["label"] == "multi" else 0, dtype=torch.long),
        }


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("label") not in {"single", "multi"}:
                raise ValueError(f"非法标签：{row.get('label')}")
            rows.append(row)
    random.Random(42).shuffle(rows)
    return rows


def split_rows(rows: list[dict]):
    size = len(rows)
    train_end = int(size * 0.8)
    valid_end = int(size * 0.9)
    return rows[:train_end], rows[train_end:valid_end], rows[valid_end:]


def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["features"].to(device),
            )
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probabilities.extend(probs.tolist())
            labels.extend(batch["label"].numpy().tolist())
    predictions = [1 if value >= 0.5 else 0 for value in probabilities]
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    try:
        result["auc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        result["auc"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/complexity-classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()
    rows = load_rows(args.data)
    if len(rows) < 100:
        raise SystemExit("训练数据过少，建议至少 500 条人工复核样本")
    train_rows, valid_rows, test_rows = split_rows(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_loader = DataLoader(QADataset(train_rows, tokenizer), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(QADataset(valid_rows, tokenizer), batch_size=args.batch_size)
    test_loader = DataLoader(QADataset(test_rows, tokenizer), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertWithFeatures(args.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["features"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        valid = evaluate(model, valid_loader, device)
        print(json.dumps({"epoch": epoch, "loss": total_loss / max(1, len(train_loader)), "valid": valid}, ensure_ascii=False))
        if valid["f1"] > best_f1:
            best_f1 = valid["f1"]
            model.encoder.save_pretrained(output / "encoder")
            tokenizer.save_pretrained(output / "tokenizer")
            torch.save(model.state_dict(), output / "model.pt")
            (output / "metrics.json").write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = BertWithFeatures(args.base_model).to(device)
    checkpoint.load_state_dict(torch.load(output / "model.pt", map_location=device))
    test_metrics = evaluate(checkpoint, test_loader, device)
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_meta.json").write_text(json.dumps({"base_model": args.base_model, "rows": len(rows)}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"test": test_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
