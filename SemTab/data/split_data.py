import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


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


def load_data(config_file: Path, mapping_file: Path) -> pd.DataFrame:
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


def split_and_save(df: pd.DataFrame, output_dir: Path, num_folds: int, test_size: float, seed: int, save_csv: bool) -> None:
    if df.empty:
        raise RuntimeError("No valid samples were loaded.")
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = sorted(df["table_hash"].dropna().unique().tolist())
    for fold_idx in range(num_folds):
        train_hashes, temp_hashes = train_test_split(hashes, test_size=test_size, random_state=seed + fold_idx, shuffle=True)
        val_hashes, test_hashes = train_test_split(temp_hashes, test_size=0.5, random_state=seed + fold_idx, shuffle=True)
        splits = {"train": df[df["table_hash"].isin(train_hashes)].copy(), "val": df[df["table_hash"].isin(val_hashes)].copy(), "test": df[df["table_hash"].isin(test_hashes)].copy()}
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in splits.items():
            split_df.to_pickle(fold_dir / f"{name}.pkl")
            if save_csv:
                split_df.to_csv(fold_dir / f"{name}.csv", index=False, encoding="utf-8")
        labels = sorted(pd.concat([splits[name]["label"] for name in ["train", "val", "test"]], axis=0).astype(str).unique().tolist())
        save_json(fold_dir / "labels.json", labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=Path, default=Path("path/to/data_config.json"))
    parser.add_argument("--mapping_file", type=Path, default=Path("path/to/mapping.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("path/to/output_data"))
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_data(args.data_config, args.mapping_file)
    split_and_save(df, args.output_dir, args.num_folds, args.test_size, args.seed, args.save_csv)


if __name__ == "__main__":
    main()
