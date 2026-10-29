import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import BertWithFeatures
from train import QADataset, evaluate, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    model_dir = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertWithFeatures(str(model_dir / "encoder")).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    loader = DataLoader(QADataset(load_rows(args.data), tokenizer), batch_size=args.batch_size)
    print(json.dumps(evaluate(model, loader, device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
