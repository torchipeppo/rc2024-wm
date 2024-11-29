import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
import hydra
import einops
import sys
import logging
from pathlib import Path

from tokenizer import Tokenizer
from transformer_nanogpt import Transformer, TransformerConfig
from data import MarioCSVDataset, MarioRealizedDataset
import utils
import metrics

# determined all the way up here so hydra doesn't mess with it
DIR_OF_THIS_FILE = Path(__file__).resolve().parent
DATA_DIR = DIR_OF_THIS_FILE.parent / "data"

# index in this list corresponds to the token ID
# Note to self: adding new reserved tokens will require a (hopefully minor) overhaul of the tokenizer.
RESERVED_TOKENS = ["UNKNOWN"]

# deve stare necessariamente qui, non in un'altra cartella
def hydra_autohandle_derived_configs(f):
    def wrapper(conf):
        # automatic device based on availability
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # device type based on device
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        
        dataset_conf = OmegaConf.load(DATA_DIR / conf.dataset.csv_path / "config.yaml")
        for key in dataset_conf:
            print(dataset_conf.keys())
            conf.dataset[key] = dataset_conf[key]
        
        # number of tokens given buckets, and other derived transformer configs
        conf.tokenizer.rel_pos_conf.total_relative_buckets = (
            2 * (conf.tokenizer.rel_pos_conf.high_precision_positive_buckets +  # this x2 is b/c the conf only specifies the no. of buckets on the positive side
                conf.tokenizer.rel_pos_conf.mid_precision_positive_buckets +
                conf.tokenizer.rel_pos_conf.low_precision_positive_buckets + 1)
        )
        conf.transformer.vocab_size = (
            conf.tokenizer.abs_pos_conf.x_buckets * conf.tokenizer.abs_pos_conf.y_buckets +
            conf.tokenizer.rel_pos_conf.total_relative_buckets ** 2 +  # this ^2 is b/c until now we only counted one axis, and x and y have the same no. of rel. buckets
            len(RESERVED_TOKENS)
        )
        conf.transformer.num_of_blocks = conf.dataset.time_span
        conf.transformer.block_size = 2 + conf.dataset.number_of_other_robots  # i.e. ego + others + ball
        try:
            _ = conf.tokenizer.alpha_buckets  # raises an exception (and nothing happens) if it does not exist, otherwise prints the following warning
            print("!!!!!!!!!!!!!!!")
            print("!!! WARNING !!!")
            print("!!!!!!!!!!!!!!!")
            print("Looks like did need alpha_buckets after all! Are they included in the vocab_size calculation?")
            print("!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!")
        except:
            pass
        
        f(conf)
    
    return wrapper


def print_statsstuff(data_list, name):
    logger = logging.getLogger(__name__)
    logger.info(name)
    logger.info(f"    mean: {np.mean(data_list)}")
    logger.info(f"    SEM: {np.std(data_list, ddof=1) / np.sqrt(np.size(data_list))}")
    logger.info(f"    variance: {np.var(data_list)}")
    logger.info(f"    median: {np.median(data_list)}")
    logger.info("")

@hydra.main(config_path="config", config_name="test-set")
@hydra_autohandle_derived_configs
def main(conf):
    device = torch.device(conf.device)
    cpu = torch.device("cpu")
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info(" ".join(sys.argv))
    logger.info("================================")
    
    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer, reserved_tokens=RESERVED_TOKENS)
    tokenizer.set_device(device)
    
    transformer = Transformer(hydra.utils.instantiate(conf.transformer), reserved_tokens=RESERVED_TOKENS).to(device)
    ckpt_path = (DIR_OF_THIS_FILE / conf.ckpt_fname).resolve()
    checkpoint = torch.load(ckpt_path, map_location=device)
    transformer.load_state_dict(checkpoint['transformer'])
    transformer.eval()
    
    # dataset
    # dataset : MarioCSVDataset = hydra.utils.instantiate(conf.dataset)
    dataset = MarioRealizedDataset(conf.dataset.csv_path)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)

    # metrics accumulation
    field_distances_list = []
    abs_field_distances_list = []
    rel_field_distances_list = []
    grid_distances_list = []
    abs_grid_distances_list = []
    rel_grid_distances_list = []

    batch_idx = -1  # solely so that I can increment at the beginning of the loop
    for batch_input, batch_target in dataloader:
        batch_idx += 1
        print(f"test_set batch {batch_idx} / {len(dataloader)}")
        
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        tokenized_input = tokenizer.tokenize(batch_input)
        tokenized_target = tokenizer.tokenize(batch_target)
        
        tokenized_input = einops.rearrange(tokenized_input, "batch time object -> batch (time object)")
        tokenized_target = einops.rearrange(tokenized_target, "batch time object -> batch (time object)")
        
        logits, _ = transformer(tokenized_input, tokenized_target)
        pred_probs = F.softmax(logits, dim=-1)
        pred_tokens = torch.topk(pred_probs, 1).indices.squeeze()
        
        # Note: they are both discretized, but grid is just about the spaces on the grid
        # while field is the center of each grid cell (in millimeters)
        pred_grid_pos, pred_field_pos = tokenizer.token_to_buckets(pred_tokens)
        targ_grid_pos, _ = tokenizer.token_to_buckets(tokenized_target)
        true_field_pos = einops.rearrange(batch_target, "batch time object coords -> batch (time object) coords")
        abs_token_mask = tokenized_target - tokenizer.num_reserved_tokens < tokenizer.x_buckets*tokenizer.y_buckets
                            # TODO tokenizer.token_is_abs
                            # i.e. trasformare la condizione enunciata per esteso qui sopra
                            # in una funzione chiamabile del tokenizer,
                            # per leggibilità/modularità/etc
                            # (i.e. non è una feature mancante, solo pulizia/stile)
        # euclidean distance of field positions
        # don't count out-of-range predictions in this particular metric
        pfp = torch.where(pred_field_pos.isinf(), np.nan, pred_field_pos)
        field_distance = metrics.field_distance(pfp, true_field_pos, abs_token_mask)
        field_distances_list.append(field_distance.glob.item())
        abs_field_distances_list.append(field_distance.abs.item())
        rel_field_distances_list.append(field_distance.rel.item())
        # manhattan distance in grid space
        grid_distance = metrics.grid_distance(pred_grid_pos, targ_grid_pos, abs_token_mask)
        grid_distances_list.append(grid_distance.glob.item())
        abs_grid_distances_list.append(grid_distance.abs.item())
        rel_grid_distances_list.append(grid_distance.rel.item())
        
        # eventuali altre robe
    
    print_statsstuff(field_distances_list, "field_distance")
    print_statsstuff(abs_field_distances_list, "abs_field_distance")
    print_statsstuff(rel_field_distances_list, "rel_field_distance")
    print_statsstuff(grid_distances_list, "grid_distance")
    print_statsstuff(abs_grid_distances_list, "abs_grid_distance")
    print_statsstuff(rel_grid_distances_list, "rel_grid_distance")


if __name__ == "__main__":
    main()
