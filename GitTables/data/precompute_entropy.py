import argparse
import concurrent.futures
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from tqdm import tqdm


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="path/to/input_data")
    parser.add_argument("--label_file", type=str, default="path/to/label_frequencies.json")
    parser.add_argument("--model_path", type=str, default="path/to/all-MiniLM-L6-v2")
    parser.add_argument("--output_file", type=str, default="entropy_lookup.pkl")
    parser.add_argument("--min_label_count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def load_labels(label_file, min_label_count):
    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        labels = [str(label) for label, count in data.items() if count >= min_label_count]
    elif isinstance(data, list):
        labels = [str(label) for label in data]
    else:
        raise ValueError("label_file must contain a JSON object or a JSON list")
    labels = sorted(set(label for label in labels if label))
    if not labels:
        raise ValueError("No valid labels found")
    return labels


def find_input_files(input_path):
    root = Path(input_path)
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {root.suffix}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No supported input files found under: {input_path}")
    return files


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path, orient="records")
    except ValueError:
        return pd.read_json(path)


def get_unique_texts_from_file(path):
    try:
        df = read_table(Path(path))
        if df.empty:
            return set()
        values = df.stack(dropna=True).astype(str)
        return set(value for value in values if value)
    except Exception:
        return set()


def collect_unique_texts(files, num_workers):
    unique_texts = set()
    if num_workers <= 1:
        iterator = map(get_unique_texts_from_file, files)
        for texts in tqdm(iterator, total=len(files), desc="Collecting texts"):
            unique_texts.update(texts)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            iterator = executor.map(get_unique_texts_from_file, files)
            for texts in tqdm(iterator, total=len(files), desc="Collecting texts"):
                unique_texts.update(texts)
    if not unique_texts:
        raise ValueError("No valid text values found in the input data")
    return sorted(unique_texts)


def compute_entropy_map(texts, labels, model_path, batch_size, temperature):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    label_embeddings = model.encode(labels, convert_to_tensor=True, show_progress_bar=False)
    label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
    entropy_map = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="Computing entropy"):
        batch_texts = texts[start:start + batch_size]
        with torch.no_grad():
            embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            logits = torch.matmul(embeddings, label_embeddings.T)
            probabilities = F.softmax(logits / temperature, dim=-1)
            entropies = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=-1).detach().cpu().tolist()
        entropy_map.update({text: float(entropy) for text, entropy in zip(batch_texts, entropies)})
    return entropy_map


def save_pickle(obj, output_file):
    output_path = Path(output_file)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    labels = load_labels(args.label_file, args.min_label_count)
    files = find_input_files(args.input_path)
    texts = collect_unique_texts(files, args.num_workers)
    entropy_map = compute_entropy_map(
        texts=texts,
        labels=labels,
        model_path=args.model_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    save_pickle(entropy_map, args.output_file)


if __name__ == "__main__":
    main()
