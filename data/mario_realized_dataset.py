import torch
from torch.utils.data import Dataset
from pathlib import Path

class MarioRealizedDataset(Dataset):
    def __init__(self, csv_path):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        # default to searching in the same directory as this Python file (so maybe I won't need path_constants this time)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent / csv_path
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
