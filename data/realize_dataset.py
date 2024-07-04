import torch
from pathlib import Path
from omegaconf import OmegaConf
import tqdm
import multiprocessing
from mario_csv_dataset import MarioCSVDataset

# se faccio più di una variante, teniamo la storia delle target_dir nei commenti
TARGET_DIR = Path(__file__).parent / "realized_5-300"

conf = OmegaConf.load(TARGET_DIR / "config.yaml")

dataset = MarioCSVDataset(
    csv_path=conf.orig_csv_path,
    number_of_other_robots=conf.number_of_other_robots,
    min_time_span=conf.min_time_span,
    time_span=conf.time_span,
    time_bw_frames=conf.time_bw_frames,
)

PROCESSES = 24

def save_sample_by_index(i):
    torch.save(dataset[i], TARGET_DIR / f"{i}.data")

pool = multiprocessing.Pool(PROCESSES)
progressbar = tqdm.tqdm(total=len(dataset))
for _ in pool.imap_unordered(save_sample_by_index, range(len(dataset)), chunksize=10):
    progressbar.update()
pool.close()
pool.join()
