from torch.utils.data import Dataset
from pathlib import Path

class MarioCSVDataset(Dataset):
    def __init__(self, csv_path):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        
        meta_path = csv_path.with_suffix("meta.yaml")  # TODO questo non esisterà più, non in questa forma

        # TODO preprocessing prima
