import torch
from pathlib import Path
from omegaconf import OmegaConf
import tqdm
import multiprocessing
from mario_csv_dataset import MarioCSVDataset

# se faccio più di una variante, teniamo la storia delle target_dir nei commenti
TARGET_DIR = Path(__file__).parent / "realized_5-150"

conf = OmegaConf.load(TARGET_DIR / "config.yaml")

dataset = MarioCSVDataset(
    csv_path=conf.orig_csv_path,
    number_of_other_robots=conf.number_of_other_robots,
    min_time_span=conf.min_time_span,
    time_span=conf.time_span,
    time_bw_frames=conf.time_bw_frames,
)

PROCESSES = 56

def save_sample_by_index(i):
    sample = dataset[i]
    if (
            sample[0].shape[0] == conf.time_span and sample[0].shape[1] == conf.number_of_other_robots+2 and
            sample[1].shape[0] == conf.time_span and sample[1].shape[1] == conf.number_of_other_robots+2):
        torch.save(sample, TARGET_DIR / f"{i}.data")
    else:
        print("Skipped shape", sample[0].shape, sample[1].shape)

pool = multiprocessing.Pool(PROCESSES)
progressbar = tqdm.tqdm(total=len(dataset))
for _ in pool.imap_unordered(save_sample_by_index, range(len(dataset)), chunksize=10):
    progressbar.update()
pool.close()
pool.join()
