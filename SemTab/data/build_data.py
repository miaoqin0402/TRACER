import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import pickle
import random
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
WORKER_ENTROPY_DICT: Dict[str, float] = {}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_file_md5(file_path: Path) -> Optional[str]:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, dtype=str, keep_default_na=False)
    except Exception:
        return pd.read_csv(path, low_memory=False, keep_default_na=False)


def normalize_cell(value) -> str:
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def verbalize_label(label) -> str:
    text = str(label).strip().rstrip("/")
    text = text.split("/")[-1].split("#")[-1]
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text if text else str(label)


def read_gt_fallback(gt_path: Path) -> pd.DataFrame:
    rows: List[List[str]] = []
    with open(gt_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts or all(str(x).strip() == "" for x in parts):
                continue
            if len(parts) >= 4:
                rows.append([str(x).strip() for x in parts[:4]])
    return pd.DataFrame(rows, columns=[0, 1, 2, 3])


def read_gt_file(gt_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(gt_path, header=None, usecols=[0, 1, 2, 3], dtype=str, engine="python", keep_default_na=False, on_bad_lines="skip")
    except TypeError:
        try:
            df = pd.read_csv(gt_path, header=None, usecols=[0, 1, 2, 3], dtype=str, engine="python", keep_default_na=False, error_bad_lines=False, warn_bad_lines=False)
        except Exception:
            df = read_gt_fallback(gt_path)
    except Exception:
        df = read_gt_fallback(gt_path)
    if df is None or df.empty:
        return pd.DataFrame(columns=[0, 1, 2, 3])
    df = df.iloc[:, :4].copy()
    df.columns = [0, 1, 2, 3]
    for col in [0, 1, 2, 3]:
        df[col] = df[col].astype(str).map(lambda x: x.strip())
    df = df[(df[0] != "") & (df[1] != "") & (df[2] != "") & (df[3] != "")]
    return df.reset_index(drop=True)


def load_configs(config_file: Path) -> List[Dict[str, object]]:
    configs = load_json(config_file)
    if not isinstance(configs, list):
        raise ValueError("The data configuration file must contain a list of dataset configurations.")
    for cfg in configs:
        for key in ["name", "gt_path", "table_dir", "need_mapping"]:
            if key not in cfg:
                raise KeyError(f"Missing key in data configuration: {key}")
    return configs


def load_data_and_merge(config_file: Path, mapping_file: Path) -> pd.DataFrame:
    configs = load_configs(config_file)
    requires_mapping = any(bool(cfg.get("need_mapping")) for cfg in configs)
    mapping = load_json(mapping_file) if requires_mapping else {}
    samples: List[Dict[str, object]] = []
    for cfg in configs:
        gt_path = Path(str(cfg["gt_path"]))
        table_dir = Path(str(cfg["table_dir"]))
        if not gt_path.exists() or not table_dir.exists():
            continue
        gt_df = read_gt_file(gt_path)
        if gt_df.empty:
            continue
        for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc=str(cfg["name"])):
            table_id = str(row[0]).strip()
            raw_label = str(row[3]).strip()
            label = raw_label
            if bool(cfg.get("need_mapping")):
                key = raw_label.rstrip("/").split("/")[-1]
                if key not in mapping:
                    continue
                label = mapping[key]
            file_name = table_id if table_id.endswith(".csv") else f"{table_id}.csv"
            table_path = table_dir / file_name
            if not table_path.exists():
                continue
            table_hash = get_file_md5(table_path)
            if table_hash is None:
                continue
            try:
                subj_idx = int(row[1])
                obj_idx = int(row[2])
            except Exception:
                continue
            samples.append({"table_id": table_id, "table_path": str(table_path), "table_hash": table_hash, "subj_col_idx": subj_idx, "obj_col_idx": obj_idx, "label": str(label), "source_dataset": str(cfg["name"])})
    return pd.DataFrame(samples)


def extract_unique_texts_from_csv(csv_path: str) -> set:
    try:
        df = read_csv_robust(Path(csv_path))
        values = df.values.ravel()
        return {normalize_cell(v) for v in values if normalize_cell(v)}
    except Exception:
        return set()


def precompute_entropy(df: pd.DataFrame, entropy_file: Path, num_workers: int, model_name: str, temperature: float, use_verbalized_labels: bool, chunk_size: int, encode_batch_size: int) -> None:
    labels = sorted(df["label"].astype(str).unique().tolist())
    label_texts = [verbalize_label(x) for x in labels] if use_verbalized_labels else labels
    table_paths = sorted(df["table_path"].astype(str).unique().tolist())
    unique_texts = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for values in tqdm(executor.map(extract_unique_texts_from_csv, table_paths), total=len(table_paths), desc="Collecting cells"):
            unique_texts.update(values)
    unique_list = sorted(unique_texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)
    label_embeddings = model.encode(label_texts, convert_to_tensor=True, show_progress_bar=False)
    label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
    entropy_map: Dict[str, float] = {}
    for start in tqdm(range(0, len(unique_list), chunk_size), desc="Computing entropy"):
        texts = unique_list[start:start + chunk_size]
        with torch.no_grad():
            embeddings = model.encode(texts, batch_size=encode_batch_size, convert_to_tensor=True, show_progress_bar=False)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            logits = torch.matmul(embeddings, label_embeddings.T)
            probs = F.softmax(logits / temperature, dim=-1)
            entropies = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).cpu().numpy()
            entropy_map.update({text: float(entropy) for text, entropy in zip(texts, entropies)})
    entropy_file.parent.mkdir(parents=True, exist_ok=True)
    with open(entropy_file, "wb") as f:
        pickle.dump(entropy_map, f, protocol=pickle.HIGHEST_PROTOCOL)


def init_worker(entropy_file: str, seed: int) -> None:
    global WORKER_ENTROPY_DICT
    random.seed(seed)
    np.random.seed(seed)
    with open(entropy_file, "rb") as f:
        WORKER_ENTROPY_DICT = pickle.load(f)


def row_entropy(df: pd.DataFrame, indices: List[int], default_entropy: float, entropy_scope: str, subj_idx: int, obj_idx: int) -> np.ndarray:
    columns = [subj_idx, obj_idx] if entropy_scope == "pair" else list(range(df.shape[1]))
    columns = [col for col in columns if 0 <= col < df.shape[1]]
    sums = np.zeros(len(indices), dtype=np.float64)
    for col_idx in columns:
        values = df.iloc[:, col_idx].map(lambda x: WORKER_ENTROPY_DICT.get(normalize_cell(x), default_entropy)).values
        sums += values[indices]
    return sums


def retrieve_rows(df: pd.DataFrame, k: int, default_entropy: float, head_rows: int, entropy_ratio: float, entropy_scope: str, subj_idx: int, obj_idx: int) -> pd.DataFrame:
    total_rows = len(df)
    if total_rows <= k:
        return df
    head_count = min(head_rows, k, total_rows)
    head_indices = list(range(head_count))
    pool_indices = [idx for idx in range(total_rows) if idx not in head_indices]
    if not pool_indices:
        return df.iloc[head_indices]
    entropy_values = row_entropy(df, pool_indices, default_entropy, entropy_scope, subj_idx, obj_idx)
    sorted_pool = [idx for _, idx in sorted(zip(entropy_values, pool_indices), key=lambda x: x[0])]
    remaining = max(0, k - len(head_indices))
    entropy_count = min(int(round(k * entropy_ratio)), remaining, len(sorted_pool))
    entropy_indices = sorted_pool[:entropy_count]
    selected = set(head_indices + entropy_indices)
    random_pool = [idx for idx in range(total_rows) if idx not in selected]
    random_count = max(0, k - len(selected))
    if random_count > 0 and random_pool:
        selected.update(random.sample(random_pool, min(random_count, len(random_pool))))
    return df.iloc[sorted(selected)]


def serialize_pairs(df: pd.DataFrame, subj_idx: int, obj_idx: int, row_token: str) -> str:
    rows: List[str] = []
    for _, row in df.iterrows():
        subj = normalize_cell(row.iloc[subj_idx]) if 0 <= subj_idx < len(row) else ""
        obj = normalize_cell(row.iloc[obj_idx]) if 0 <= obj_idx < len(row) else ""
        if subj and obj:
            rows.append(f"{row_token} subject: {subj} ; object: {obj}")
    return " ".join(rows)


def process_sample(sample: Dict[str, object], k: int, default_entropy: float, head_rows: int, entropy_ratio: float, entropy_scope: str, row_token: str) -> Optional[Dict[str, object]]:
    try:
        table_path = Path(str(sample["table_path"]))
        if not table_path.exists():
            return None
        df = read_csv_robust(table_path)
        subj_idx = int(sample["subj_col_idx"])
        obj_idx = int(sample["obj_col_idx"])
        if subj_idx < 0 or obj_idx < 0 or subj_idx >= df.shape[1] or obj_idx >= df.shape[1]:
            return None
        selected_df = retrieve_rows(df, k, default_entropy, head_rows, entropy_ratio, entropy_scope, subj_idx, obj_idx)
        pair_text = serialize_pairs(selected_df, subj_idx, obj_idx, row_token)
        if not pair_text:
            return None
        subj_data = " ".join(normalize_cell(x) for x in selected_df.iloc[:, subj_idx].tolist() if normalize_cell(x))
        obj_data = " ".join(normalize_cell(x) for x in selected_df.iloc[:, obj_idx].tolist() if normalize_cell(x))
        return {"table_id": sample["table_id"], "table_path": str(table_path), "table_hash": sample["table_hash"], "subj_col_idx": subj_idx, "obj_col_idx": obj_idx, "pair_text": pair_text, "subj_data": subj_data, "obj_data": obj_data, "label": str(sample["label"]), "source_dataset": str(sample["source_dataset"]), "num_retrieved_rows": int(len(selected_df))}
    except Exception:
        return None


def process_split(df: pd.DataFrame, executor: concurrent.futures.ProcessPoolExecutor, worker_func) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    for result in tqdm(executor.map(worker_func, df.to_dict("records"), chunksize=100), total=len(df)):
        if result is not None:
            results.append(result)
    return pd.DataFrame(results)


def build_dataset(df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_hashes = sorted(df["table_hash"].dropna().unique().tolist())
    worker_func = partial(process_sample, k=args.k, default_entropy=args.default_entropy, head_rows=args.head_rows, entropy_ratio=args.entropy_ratio, entropy_scope=args.entropy_scope, row_token=args.row_token)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(str(args.entropy_file), args.seed)) as executor:
        for fold_idx in range(args.num_folds):
            train_hashes, temp_hashes = train_test_split(unique_hashes, test_size=args.test_size, random_state=args.seed + fold_idx, shuffle=True)
            val_hashes, test_hashes = train_test_split(temp_hashes, test_size=0.5, random_state=args.seed + fold_idx, shuffle=True)
            split_frames = {
                "train": df[df["table_hash"].isin(train_hashes)],
                "val": df[df["table_hash"].isin(val_hashes)],
                "test": df[df["table_hash"].isin(test_hashes)],
            }
            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            built_frames = {}
            for split_name, split_df in split_frames.items():
                built = process_split(split_df, executor, worker_func)
                built.to_pickle(fold_dir / f"{split_name}.pkl")
                built_frames[split_name] = built
            labels = sorted(pd.concat([built_frames[name]["label"] for name in ["train", "val", "test"] if not built_frames[name].empty], axis=0).astype(str).unique().tolist())
            save_json(fold_dir / "labels.json", labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["entropy", "build", "all"], required=True)
    parser.add_argument("--data_config", type=Path, default=Path("path/to/data_config.json"))
    parser.add_argument("--mapping_file", type=Path, default=Path("path/to/mapping.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("path/to/output_data"))
    parser.add_argument("--entropy_file", type=Path, default=Path("path/to/entropy.pkl"))
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--head_rows", type=int, default=2)
    parser.add_argument("--entropy_ratio", type=float, default=0.7)
    parser.add_argument("--entropy_scope", choices=["all", "pair"], default="pair")
    parser.add_argument("--default_entropy", type=float, default=2.0)
    parser.add_argument("--row_token", type=str, default="[ROW]")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sentence_model_name", type=str, default="path/to/sentence_transformer")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--use_raw_labels", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--encode_batch_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    df = load_data_and_merge(args.data_config, args.mapping_file)
    if df.empty:
        raise RuntimeError("No valid samples were loaded.")
    if args.step in {"entropy", "all"}:
        precompute_entropy(df, args.entropy_file, args.num_workers, args.sentence_model_name, args.temperature, not args.use_raw_labels, args.chunk_size, args.encode_batch_size)
    if args.step in {"build", "all"}:
        if not Path(args.entropy_file).exists():
            raise FileNotFoundError(f"Entropy file not found: {args.entropy_file}")
        build_dataset(df, args)


if __name__ == "__main__":
    main()
