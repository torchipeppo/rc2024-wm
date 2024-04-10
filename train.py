import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import hydra
import einops
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
    
    optimizer = transformer.configure_optimizers(**(OmegaConf.to_object(conf.training.transformer)), device_type=conf.device_type)
    
    global_step = 0
    for epoch_idx in range(conf.training.epochs):
        for batch_idx in range(conf.training.batches_per_epoch):
            print(f"epoch {epoch_idx}  batch {batch_idx}", end="")
            global_step += conf.training.batch_size
            optimizer.zero_grad()
            
            if batch_idx == conf.training.batches_per_epoch-1:
                eval_mode = True
                batch_input, batch_target = eval_iterator.next()
                transformer.eval()
                print(" (eval)")
            else:
                eval_mode = False
                batch_input, batch_target = train_iterator.next()
                transformer.train()
                print()
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            tokenized_input = tokenizer.tokenize(batch_input)
            tokenized_target = tokenizer.tokenize(batch_target)
            
            tokenized_input = einops.rearrange(tokenized_input, "batch time object -> batch (time object)")
            tokenized_target = einops.rearrange(tokenized_target, "batch time object -> batch (time object)")
            
            # TODO rivedere dataset e/o tokenizzazione: gli oggetti quasi non si spostano dalla propria cella :)
            #      posso fare due cose: o prendo frame a intervalli un po' più lunghi (tipo 1 secondo)  [forse meglio]
            #      o prendo un intorno dell'ego-robot, anziché tutto il campo
            
            logits, loss = transformer(tokenized_input, tokenized_target)
    
    # TODO continuare col ciclo di training


if __name__ == "__main__":
    main()
