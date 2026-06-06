from pathlib import Path
from datasets import load_dataset, load_from_disk

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "squad_v2"

def save_data():
    dataset = load_dataset("squad_v2")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(DATA_DIR)
    return dataset

def load_data():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Run save_data() first to download it.")
    return load_from_disk(DATA_DIR)

save_data()