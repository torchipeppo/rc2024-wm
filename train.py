import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
import sys
import logging

from tokenizer import Tokenizer
from transformer_nanogpt import Transformer, TransformerConfig
from data import MarioCSVDataset
import utils

# deve stare necessariamente qui, non in un'altra cartella
def hydra_autohandle_derived_configs(f):
    def wrapper(conf):
        # automatic device based on availability
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # device type based on device
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        
        # number of tokens given buckets, and other derived transformer configs
        conf.transformer.vocab_size = conf.tokenizer.x_buckets * conf.tokenizer.y_buckets
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


@hydra.main(config_path="config", config_name="default")
@hydra_autohandle_derived_configs
def main(conf):
    device = torch.device(conf.device)
    cpu = torch.device("cpu")
    
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info("================================")
    
    writer = SummaryWriter("")
    
    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer)
    
    transformer = Transformer(hydra.utils.instantiate(conf.transformer))
    
    # dataset
    dataset : MarioCSVDataset = hydra.utils.instantiate(conf.dataset)
    # train-test split
    datasplit_rng = torch.Generator()
    datasplit_rng.manual_seed(14383421)
    train_len = int(len(dataset) * conf.training.data_split)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=datasplit_rng)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=True, drop_last=True)
    train_iterator = utils.infiniter(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=conf.eval.batch_size, shuffle=True, drop_last=True)
    eval_iterator = utils.infiniter(eval_dataloader)
    
    # TODO optimizer qui
    
    # TODO continuare col ciclo di training


if __name__ == "__main__":
    main()
