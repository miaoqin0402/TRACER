import logging
import pickle
from pathlib import Path


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def save_object(obj, filepath):
    filepath = Path(filepath)
    if filepath.parent and str(filepath.parent) != ".":
        filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filepath):
    with Path(filepath).open("rb") as f:
        return pickle.load(f)


def get_data_path_from_metadata_path(metadata_path, json_data_root, metadata_root):
    metadata_path = Path(metadata_path)
    json_data_root = Path(json_data_root)
    metadata_root = Path(metadata_root)
    filename = metadata_path.name.replace("_1_metadata.json", ".json")
    filename = filename.replace("_metadata.json", ".json")
    try:
        relative_path = metadata_path.relative_to(metadata_root).parent / filename
        return json_data_root / relative_path
    except ValueError:
        return json_data_root / filename
