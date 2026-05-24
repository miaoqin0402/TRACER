import argparse
import logging
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel, get_linear_schedule_with_warmup

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


class TableColumnDataset(Dataset):
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


def collate_fn(batch, tokenizer, max_col_len, max_seq_len):
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


def load_data(data_file):
    with open(data_file, "rb") as file:
        data = pickle.load(file)
    for key in ["train", "dev", "label_encoder"]:
        if key not in data:
            raise KeyError(f"The input file must contain '{key}'.")
    return data


def evaluate(model, data_loader, device, use_amp):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_mask = batch["cls_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=use_amp):
                logits, = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = logits[cls_mask].argmax(dim=-1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.tolist())

    if not all_targets:
        return {"micro": (0.0, 0.0, 0.0), "macro": (0.0, 0.0, 0.0)}

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(all_targets, all_predictions, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_predictions, average="macro", zero_division=0)
    return {"micro": (p_micro, r_micro, f1_micro), "macro": (p_macro, r_macro, f1_macro)}


def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and device.type == "cuda"

    logging.info(f"Loading dataset from {args.data_file}")
    data = load_data(args.data_file)
    label_encoder = data["label_encoder"]
    num_labels = len(label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = TableColumnDataset(data["train"], label_encoder)
    dev_dataset = TableColumnDataset(data["dev"], label_encoder)
    collate = partial(collate_fn, tokenizer=tokenizer, max_col_len=args.max_col_len, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
    model = DistilBertForColumnClassification.from_pretrained(args.model_name, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=use_amp)
    loss_fn = CrossEntropyLoss()
    best_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        update_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_mask = batch["cls_mask"].to(device)
            labels = batch["labels"].to(device)
            if labels.numel() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits, = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits[cls_mask], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            update_steps += 1

        metrics = evaluate(model, dev_loader, device, use_amp)
        avg_loss = total_loss / max(1, update_steps)
        p_micro, r_micro, f1_micro = metrics["micro"]
        p_macro, r_macro, f1_macro = metrics["macro"]
        logging.info(f"Epoch {epoch + 1} | Loss: {avg_loss:.6f}")
        logging.info(f"Micro | F1: {f1_micro:.6f} | Precision: {p_micro:.6f} | Recall: {r_micro:.6f}")
        logging.info(f"Macro | F1: {f1_macro:.6f} | Precision: {p_macro:.6f} | Recall: {r_macro:.6f}")

        if f1_micro > best_score:
            best_score = f1_micro
            output_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({"model_state_dict": model.state_dict(), "num_labels": num_labels, "label_classes": list(label_encoder.classes_), "best_micro_f1": best_score}, output_path)
            logging.info(f"Saved best model to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="path/to/dataset.pkl")
    parser.add_argument("--output_dir", type=str, default="path/to/checkpoints")
    parser.add_argument("--model_name", type=str, default="path/to/pretrained_model")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_col_len", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
