import argparse
import logging
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel, BertPreTrainedModel, DistilBertModel, DistilBertPreTrainedModel

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DistilBertForColumnClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        return (logits,)


class BertForColumnClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        return (logits,)


def get_model_class(config):
    if config.model_type == "distilbert":
        return DistilBertForColumnClassification
    if config.model_type == "bert":
        return BertForColumnClassification
    raise ValueError(f"Unsupported model type: {config.model_type}")


class ColumnClassificationDataset(Dataset):
    def __init__(self, dataframe, label_encoder):
        required_columns = {"table_id", "data", "label"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        self.groups = list(dataframe.groupby("table_id"))
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        _, group = self.groups[index]
        texts = group["data"].fillna("").astype(str).tolist()
        labels = torch.as_tensor(self.label_encoder.transform(group["label"].astype(str).tolist()), dtype=torch.long)
        return {"texts": texts, "labels": labels}


def collate_fn(batch, tokenizer, max_col_len=32):
    max_seq_len = 512
    batch_input_ids = []
    batch_attention_mask = []
    batch_cls_mask = []
    batch_labels = []
    effective_col_len = max(1, min(max_col_len + 2, max_seq_len))

    for item in batch:
        packed_ids = []
        cls_mask = []
        valid_labels = []

        for text, label in zip(item["texts"], item["labels"]):
            ids = tokenizer.encode(text, add_special_tokens=True, max_length=effective_col_len, truncation=True)
            if not ids:
                continue
            if len(packed_ids) + len(ids) > max_seq_len:
                if packed_ids:
                    break
                ids = ids[:max_seq_len]
            current_cls_mask = [False] * len(ids)
            current_cls_mask[0] = True
            packed_ids.extend(ids)
            cls_mask.extend(current_cls_mask)
            valid_labels.append(label)

        if not packed_ids or not valid_labels:
            continue

        batch_input_ids.append(torch.as_tensor(packed_ids, dtype=torch.long))
        batch_attention_mask.append(torch.ones(len(packed_ids), dtype=torch.long))
        batch_cls_mask.append(torch.as_tensor(cls_mask, dtype=torch.bool))
        batch_labels.append(torch.stack(valid_labels))

    if not batch_input_ids:
        return None

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0),
        "cls_mask": torch.nn.utils.rnn.pad_sequence(batch_cls_mask, batch_first=True, padding_value=False),
        "labels": torch.cat(batch_labels, dim=0),
    }


def load_data(data_pkl):
    with open(data_pkl, "rb") as file:
        data = pickle.load(file)
    for key in ["test", "label_encoder"]:
        if key not in data:
            raise KeyError(f"The input file must contain '{key}'.")
    return data


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model


def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_mask = batch["cls_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=device.type == "cuda"):
                logits, = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = logits[cls_mask].argmax(dim=-1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.tolist())

    if not all_targets:
        return {
            "accuracy": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "num_samples": 0,
        }

    accuracy = accuracy_score(all_targets, all_predictions)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average="micro", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "num_samples": len(all_targets),
    }


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_pkl)
    label_encoder = data["label_encoder"]
    num_labels = len(label_encoder.classes_)

    config = AutoConfig.from_pretrained(args.reader_model_name, num_labels=num_labels)
    if args.use_flash_attention:
        config.attn_implementation = "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    model_class = get_model_class(config)
    model = model_class.from_pretrained(args.reader_model_name, config=config)
    model = load_checkpoint(model, args.reader_checkpoint, device).to(device)

    dataset = ColumnClassificationDataset(data["test"], label_encoder)
    data_collator = partial(collate_fn, tokenizer=tokenizer, max_col_len=args.max_len_per_col)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    metrics = evaluate(model, data_loader, device)
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"Micro-Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro-Recall: {metrics['micro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall: {metrics['macro_recall']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pkl", type=str, default="path/to/dataset.pkl")
    parser.add_argument("--reader_model_name", type=str, default="path/to/pretrained_model")
    parser.add_argument("--reader_checkpoint", type=str, default="path/to/best_model.pt")
    parser.add_argument("--max_len_per_col", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_flash_attention", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
