import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class PairDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, label2id: Dict[str, int], max_length: int, keep_text: bool = False):
        self.df = pd.read_pickle(data_path)
        required = {"pair_text", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise KeyError(f"{data_path} missing required columns: {sorted(missing)}")
        self.df = self.df.dropna(subset=["pair_text", "label"]).reset_index(drop=True)
        self.df["pair_text"] = self.df["pair_text"].astype(str)
        self.df["label"] = self.df["label"].astype(str)
        self.df = self.df[self.df["pair_text"].str.len() > 0].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.keep_text = keep_text

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        label = str(row["label"])
        if label not in self.label2id:
            raise KeyError(f"Label not found in labels.json: {label}")
        encoding = self.tokenizer(str(row["pair_text"]), max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        item = {"input_ids": encoding["input_ids"].flatten(), "attention_mask": encoding["attention_mask"].flatten(), "labels": torch.tensor(self.label2id[label], dtype=torch.long), "idx": idx}
        if self.keep_text:
            item["pair_text"] = str(row["pair_text"])
        return item


class PairClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float, use_flash_attention: bool):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        kwargs = {}
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.encoder = AutoModel.from_pretrained(model_name, **kwargs)
        hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "dim", None))
        if hidden_size is None:
            raise ValueError("Cannot infer encoder hidden size from model config.")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))


def load_labels(data_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    labels_path = data_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.json: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [str(x) for x in json.load(f)]
    return labels, {label: idx for idx, label in enumerate(labels)}


def load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(checkpoint_path, map_location="cpu")
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"acc": float(acc), "micro_precision": float(p_micro), "micro_recall": float(r_micro), "micro_f1": float(f1_micro), "macro_precision": float(p_macro), "macro_recall": float(r_macro), "macro_f1": float(f1_macro), "weighted_precision": float(p_weighted), "weighted_recall": float(r_weighted), "weighted_f1": float(f1_weighted)}


def main(args: argparse.Namespace) -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    labels, label2id = load_labels(data_dir)
    split_path = data_dir / f"{args.split}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = PairDataset(split_path, tokenizer, label2id, args.max_length, keep_text=args.save_predictions is not None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = PairClassifier(args.model_name, len(labels), args.dropout, args.use_flash_attention)
    state = load_state_dict(Path(args.checkpoint_path))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_indices: List[int] = []
    all_texts: List[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)
            if device.type == "cuda" and args.use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels_tensor.cpu().numpy().tolist())
            all_indices.extend(batch["idx"].cpu().numpy().tolist())
            if args.save_predictions is not None:
                all_texts.extend(batch["pair_text"])
    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({"split": args.split, "num_samples": len(all_labels), "max_length": args.max_length, "batch_size": args.batch_size})
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.print_class_report:
        print(classification_report(all_labels, all_preds, labels=list(range(len(labels))), target_names=labels, zero_division=0, digits=4))
    if args.save_summary is not None:
        save_json(Path(args.save_summary), metrics)
    if args.save_predictions is not None:
        out_path = Path(args.save_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"idx": all_indices, "gold_id": all_labels, "pred_id": all_preds, "gold_label": [labels[i] for i in all_labels], "pred_label": [labels[i] for i in all_preds], "correct": [int(g == p) for g, p in zip(all_labels, all_preds)], "pair_text": all_texts if all_texts else [""] * len(all_labels)}).to_csv(out_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="path/to/fold")
    parser.add_argument("--checkpoint_path", type=str, default="path/to/best_model.pt")
    parser.add_argument("--model_name", type=str, default="path/to/pretrained_model")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--print_class_report", action="store_true")
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument("--save_summary", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
