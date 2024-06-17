import torch

# angle-less version
# does MARIO even give us angles?
class Tokenizer:
    # x_buckets also counts the two "special" buckets for when x<x_min or x>x_max.
    # this is because the number of possible outcomes of the bucketize function
    # is directly correlated to the number of possible tokens for the transformer,
    # so I need to be precise.
    # same for y_buckets
    def __init__(self, x_min, y_min, x_max, y_max, x_buckets, y_buckets, reserved_tokens):
        assert x_min < x_max
        assert y_min < y_max
        assert x_buckets > 2, 'Need at least 2 buckets for "out of range"'
        assert y_buckets > 2, 'Need at least 2 buckets for "out of range"'
        self.x_buckets = x_buckets
        self.y_buckets = y_buckets
        
        # given the number of buckets, I want that a number of boundaries equal to buckets-1
        # so that I get buckets-2 "internal" buckets, plus 2 "out-of-range" buckets
        self.x_boundaries = torch.linspace(x_min, x_max, x_buckets-1)
        self.y_boundaries = torch.linspace(y_min, y_max, y_buckets-1)
        
        self.num_reserved_tokens = len(reserved_tokens)
        self.UNKNOWN = reserved_tokens.index("UNKNOWN")
    
    # data shape: [... object coords] where coords size is 2 (x y) (or 3 (x y alpha), if/when the time comes to implement that)
    def tokenize(self, data):
        x_bucketized = torch.bucketize(data, self.x_boundaries)[..., 0]
        y_bucketized = torch.bucketize(data, self.y_boundaries)[..., 1]
        
        # "reduce" coords dimension by generating a unique token ID for each combination of XY buckets,
        # kinda like a y_buckets-base number
        tokenized = x_bucketized * self.y_buckets + y_bucketized + self.num_reserved_tokens
        
        # put UNKNOWN token where data was unavailable (NaN)
        # exploit the rule that anything+nan=nan to reduce the coords dimension
        tokenized = torch.where(data.sum(dim=-1).isnan(), self.UNKNOWN, tokenized)
        
        return tokenized
    
    def set_device(self, device):
        self.x_boundaries = self.x_boundaries.to(device)
        self.y_boundaries = self.y_boundaries.to(device)


if __name__ == "__main__":
    t = Tokenizer(0, 0, 4, 3, 6, 5)
    t.tokenize(torch.Tensor([[[-1,-1], [-1,0.1], [-1,1.1], [-1,2.2], [-1,3.3]],
                             [[0.1,-1], [0.1,0.1], [0.1,1.1], [0.1,2.2], [0.1,3.3]],
                             [[1.1,-1], [1.1,0.1], [1.1,1.1], [1.1,2.2], [1.1,3.3]],
                             [[2.2,-1], [2.2,0.1], [2.2,1.1], [2.2,2.2], [2.2,3.3]],
                             [[3.3,-1], [3.3,0.1], [3.3,1.1], [3.3,2.2], [3.3,3.3]],
                             [[4.4,-1], [4.4,0.1], [4.4,1.1], [4.4,2.2], [4.4,3.3]]]))
