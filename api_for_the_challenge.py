import torch
import torch.nn.functional as F
import numpy as np
import hydra
import einops

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

    def predict(self, state_history):
        # TODO aspettando gli altri
        print("WARNING: actually processing state_history is not implemented yet!!")
        state_history =  torch.tensor([[[-3993.4368,  -614.3245],
                                        [  397.6309,    72.5317],
                                        [  489.2399,  -294.1873],
                                        [ 3420.3403,  -491.7186]],
                                       [[-3981.3906,  -602.9895],
                                        [  531.0874,   216.4618],
                                        [  490.1821,  -334.4461],
                                        [ 4439.5122,   199.6088]],
                                       [[-3980.1685,  -560.0912],
                                        [  679.4299,   176.0132],
                                        [  476.9653,  -260.9104],
                                        [ 5601.6743,  1850.9146]],
                                      [[-3979.2319,  -557.6035],
                                        [ 1874.2789,    30.2208],
                                        [  404.2251,    34.8296],
                                        [    np.nan,     np.nan]],
                                      [[-3964.3123,  -540.9557],
                                        [ 1886.9766,    16.2648],
                                        [  332.7610,   258.2531],
                                        [    np.nan,     np.nan]]])
        
        tokenized_input = self.tokenizer.tokenize(state_history)
        tokenized_input = einops.rearrange(tokenized_input, "time object -> 1 (time object)")

        logits, _ = self.transformer(tokenized_input)
        pred_probs = F.softmax(logits, dim=-1)
        pred_tokens = torch.topk(pred_probs, 1).indices.squeeze()  # also squeezes out the unitary batch dim
        _, pred_field_pos = self.tokenizer.token_to_buckets(pred_tokens)

        return pred_field_pos


# TEST
if __name__ == "__main__":
    wm = TorchipeppoWorldModel("outputs/2024-07-02/12-37-01/rc2024-wm.pt")
    print(wm.predict(None))
