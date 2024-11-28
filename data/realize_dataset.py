import torch
from pathlib import Path
import shutil
from omegaconf import OmegaConf
import tqdm
import multiprocessing
from mario_csv_dataset import MarioCSVDataset
import argparse
import numpy as np

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# se faccio più di una variante, teniamo la storia dei realized_dataset_name nei commenti
REALIZED_DATASET_NAME = "realized_SAC2024"

TARGET_DIR = DATA_DIR / REALIZED_DATASET_NAME
CONF_PATH = Path(__file__).parent / f"{REALIZED_DATASET_NAME}.conf.yaml"

conf = OmegaConf.load(CONF_PATH)

if TARGET_DIR.exists():
    print("WARNING: The following target directory already exists:")
    print(f"    {TARGET_DIR}")
    print("Continuing will cause you to LOSE THE WHOLE DATASET contained in that folder.")
    print("ARE YOU SURE this is what you really want? Enter YES to confirm, NO to cancel and review the TARGET_DIR:")
    ans = input(" > ")
    if ans.upper() == "YES":
        shutil.rmtree(TARGET_DIR)
        # and continue the script below
    else:
        exit()

TARGET_DIR.mkdir()
shutil.copy(CONF_PATH, TARGET_DIR / "config.yaml")
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

# "test set mode" that fill axis 1 with nan
# until shape[1] == conf.number_of_other_robots+2,
# so I have more test samples.
# I can afford this for a test dataset,
# since I need full data for training.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test-set-mode', action='store_true')
args = parser.parse_args()

def save_sample_by_index(i):
    sample = dataset[i]
    assert sample[0].shape == sample[1].shape
    if args.test_set_mode:
        if sample[0].shape[1] < conf.number_of_other_robots+2:
            extra_cols = torch.full(
                [
                    sample[0].shape[0],
                    conf.number_of_other_robots+2-sample[0].shape[1],
                    sample[0].shape[2],
                ],
                np.nan,
            )
            sample = (
                torch.cat((sample[0], extra_cols), dim=1),
                torch.cat((sample[1], extra_cols), dim=1)
            )
            assert sample[0].shape[1] == conf.number_of_other_robots+2, (sample[0].shape[1], conf.number_of_other_robots+2)
            assert sample[0].shape == sample[1].shape
    if (
            sample[0].shape[0] == conf.time_span and sample[0].shape[1] == conf.number_of_other_robots+2 and
            sample[1].shape[0] == conf.time_span and sample[1].shape[1] == conf.number_of_other_robots+2):
        torch.save(sample, TARGET_DIR / "data" / f"{i}.data")  # added a "data" subfolder so the config is more accessible
    else:
        assert not args.test_set_mode  # tqdm.tqdm.write(f"Skipped shape {sample[0].shape} {sample[1].shape}")

pool = multiprocessing.Pool(PROCESSES)
progressbar = tqdm.tqdm(total=len(dataset))
for _ in pool.imap_unordered(save_sample_by_index, range(len(dataset)), chunksize=10):
    progressbar.update()
pool.close()
pool.join()
progressbar.close()
