import torch
import hydra

from tokenizer import Tokenizer
from transformer_nanogpt import Transformer

class TorchipeppoWorldModel:  # the name "torchipeppo" has no relation to pytorch btw

    def __init__(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        conf = checkpoint["conf"]
        reserved_tokens = checkpoint["reserved_tokens"]

        self.tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer, reserved_tokens=reserved_tokens)
        self.transformer = Transformer(hydra.utils.instantiate(conf.transformer), reserved_tokens=reserved_tokens)
        self.transformer.load_state_dict(checkpoint["transformer"])

        # ...more?

    def predict(self):
        raise NotImplementedError()
    
    # TODO decidere se: predict prende come argomento la storia passata,
    #      o se ce la conserviamo in questa classe e aggiungiamo un metodo tipo add_real_frame
    #      o se teniamo la storia in un file su disco
