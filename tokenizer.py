import torch
from dataclasses import dataclass
import einops

@dataclass
class AbsolutePosConf:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    x_buckets: int
    y_buckets: int

@dataclass
class RelativePosConf:
    threshold_high_precision: float  # if below this, precision is high, like 10cm, up to a best of 2cm
    threshold_mid_precision: float   # if below this, precision is average, like 20cm
    threshold_low_precision: float   # if below this, precision is low, like 50cm (above this, it's just out of range)

    high_precision_positive_buckets: int
    mid_precision_positive_buckets: int
    low_precision_positive_buckets: int  # does NOT include "out-of-range" bucket (unlike AbsolutePosConf)
    total_relative_buckets: int  # redundant field, set in the inital wrapper

# angle-less version
# does MARIO even give us angles?
class Tokenizer:
    # x_buckets also counts the two "special" buckets for when x<x_min or x>x_max.
    # this is because the number of possible outcomes of the bucketize function
    # is directly correlated to the number of possible tokens for the transformer,
    # so I need to be precise.
    # same for y_buckets
    def __init__(self, abs_pos_conf:AbsolutePosConf, rel_pos_conf:RelativePosConf, reserved_tokens):
        assert abs_pos_conf.x_min < abs_pos_conf.x_max
        assert abs_pos_conf.y_min < abs_pos_conf.y_max
        assert abs_pos_conf.x_buckets > 2, 'Need at least 2 buckets for "out of range"'
        assert abs_pos_conf.y_buckets > 2, 'Need at least 2 buckets for "out of range"'
        assert rel_pos_conf.threshold_high_precision < rel_pos_conf.threshold_mid_precision
        assert rel_pos_conf.threshold_mid_precision < rel_pos_conf.threshold_low_precision
        assert rel_pos_conf.high_precision_positive_buckets > 0
        assert rel_pos_conf.mid_precision_positive_buckets > 0
        assert rel_pos_conf.low_precision_positive_buckets > 0
        self.x_buckets = abs_pos_conf.x_buckets
        self.y_buckets = abs_pos_conf.y_buckets
        self.rel_buckets = rel_pos_conf.total_relative_buckets
        
        # given the number of buckets, I want that a number of boundaries equal to buckets-1
        # so that I get buckets-2 "internal" buckets, plus 2 "out-of-range" buckets
        self.x_boundaries = torch.linspace(
            abs_pos_conf.x_min,
            abs_pos_conf.x_max,
            abs_pos_conf.x_buckets-1
        )
        self.y_boundaries = torch.linspace(
            abs_pos_conf.y_min,
            abs_pos_conf.y_max,
            abs_pos_conf.y_buckets-1
        )
        
        # here I want exactly 2*buckets "internal" buckets
        high_prec_boundaries = torch.linspace(
            -rel_pos_conf.threshold_high_precision,
            rel_pos_conf.threshold_high_precision,
            2*rel_pos_conf.high_precision_positive_buckets+1
        )
        # here I want exactly 1*buckets "internal" buckets per side
        mid_prec_positive_boundaries = torch.linspace(
            rel_pos_conf.threshold_high_precision,
            rel_pos_conf.threshold_mid_precision,
            rel_pos_conf.mid_precision_positive_buckets+1
        )
        mid_prec_negative_boundaries = torch.linspace(
            -rel_pos_conf.threshold_mid_precision,
            -rel_pos_conf.threshold_high_precision,
            rel_pos_conf.mid_precision_positive_buckets+1
        )
        # here I want exactly 1*buckets "internal" buckets AND 1 "out of range" bucket. All per side.
        # the logic doesn't change btw.
        low_prec_positive_boundaries = torch.linspace(
            rel_pos_conf.threshold_mid_precision,
            rel_pos_conf.threshold_low_precision,
            rel_pos_conf.low_precision_positive_buckets+1
        )
        low_prec_negative_boundaries = torch.linspace(
            -rel_pos_conf.threshold_low_precision,
            -rel_pos_conf.threshold_mid_precision,
            rel_pos_conf.low_precision_positive_buckets+1
        )
        # and now for the big asssembly!
        # note that the thresholds are duplicated on two adjacent pieces
        self.relative_boundaries = torch.cat((
            low_prec_negative_boundaries,
            mid_prec_negative_boundaries[1:],
            high_prec_boundaries[1:],
            mid_prec_positive_boundaries[1:],
            low_prec_positive_boundaries[1:]
        ))
        
        self.num_reserved_tokens = len(reserved_tokens)
        self.UNKNOWN = reserved_tokens.index("UNKNOWN")
    
    # data shape: [... object coords] where coords size is 2 (x y) (or 3 (x y alpha), if/when the time comes to implement that)
    def tokenize(self, data):
        x_abs_bucketized = torch.bucketize(data, self.x_boundaries)[..., 0]
        y_abs_bucketized = torch.bucketize(data, self.y_boundaries)[..., 1]
        xy_rel_bucketized = torch.bucketize(data, self.relative_boundaries)
        x_rel_bucketized = xy_rel_bucketized[..., 0]
        y_rel_bucketized = xy_rel_bucketized[..., 1]
        
        # "reduce" coords dimension by generating a unique token ID for each combination of XY buckets,
        # kinda like a y_buckets-base number
        abs_tokenized = x_abs_bucketized * self.y_buckets + y_abs_bucketized + self.num_reserved_tokens
        # assign a different range to relative positions
        rel_tokenized = x_rel_bucketized * self.rel_buckets + y_rel_bucketized + self.x_buckets*self.y_buckets + self.num_reserved_tokens
        
        tokenized = torch.cat([abs_tokenized[..., [0]], rel_tokenized[..., 1:]], dim=-1)
        
        # put UNKNOWN token where data was unavailable (NaN)
        # exploit the rule that anything+nan=nan to reduce the coords dimension
        tokenized = torch.where(data.sum(dim=-1).isnan(), self.UNKNOWN, tokenized)
        
        return tokenized
    
    def set_device(self, device):
        self.x_boundaries = self.x_boundaries.to(device)
        self.y_boundaries = self.y_boundaries.to(device)
        self.relative_boundaries = self.relative_boundaries.to(device)


if __name__ == "__main__":
    t = Tokenizer(0, 0, 4, 3, 6, 5)
    t.tokenize(torch.Tensor([[[-1,-1], [-1,0.1], [-1,1.1], [-1,2.2], [-1,3.3]],
                             [[0.1,-1], [0.1,0.1], [0.1,1.1], [0.1,2.2], [0.1,3.3]],
                             [[1.1,-1], [1.1,0.1], [1.1,1.1], [1.1,2.2], [1.1,3.3]],
                             [[2.2,-1], [2.2,0.1], [2.2,1.1], [2.2,2.2], [2.2,3.3]],
                             [[3.3,-1], [3.3,0.1], [3.3,1.1], [3.3,2.2], [3.3,3.3]],
                             [[4.4,-1], [4.4,0.1], [4.4,1.1], [4.4,2.2], [4.4,3.3]]]))
