import torch
from torch.utils.data import Dataset
from pathlib import Path

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

class MarioRealizedDataset(Dataset):
    def __init__(self, csv_path):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        # default to searching in a specific folder relative to this Python file
        # (so maybe I won't need path_constants this time)
        if not csv_path.is_absolute():
            csv_path = DATA_DIR / csv_path / "data"  # added a "data" subfolder so the config is more accessible
        self.csv_path = csv_path
        max_idx = -1
        self.fnames = []
        for fname in self.csv_path.iterdir():
            if fname.suffix == '.data':
                self.fnames.append(fname.name)
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        return torch.load(self.csv_path / self.fnames[idx])
