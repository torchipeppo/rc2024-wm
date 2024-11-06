import torch
from pathlib import Path
import shutil
from omegaconf import OmegaConf
import tqdm
import multiprocessing
from mario_csv_dataset import MarioCSVDataset

from CONSTANTS import DATA_DIR

# se faccio più di una variante, teniamo la storia dei realized_dataset_name nei commenti
REALIZED_DATASET_NAME = "realized_5-150"

TARGET_DIR = DATA_DIR / REALIZED_DATASET_NAME
CONF_PATH = Path(__file__).parent / f"{REALIZED_DATASET_NAME}.conf.yaml"

conf = OmegaConf.load(CONF_PATH)

TARGET_DIR.mkdir()
shutil.copy(CONF_PATH, TARGET_DIR)
(TARGET_DIR/"data").mkdir()

print("Loading MarioCSVDataset, this may take a while...")
dataset = MarioCSVDataset(
    csv_path=DATA_DIR/conf.orig_csv_path,
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
        torch.save(sample, TARGET_DIR / "data" / f"{i}.data")  # added a "data" subfolder so the config is more accessible
    else:
        pass  # tqdm.tqdm.write(f"Skipped shape {sample[0].shape} {sample[1].shape}")

pool = multiprocessing.Pool(PROCESSES)
progressbar = tqdm.tqdm(total=len(dataset))
for _ in pool.imap_unordered(save_sample_by_index, range(len(dataset)), chunksize=10):
    progressbar.update()
pool.close()
pool.join()
progressbar.close()
