import torch
import numpy as np
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
        abs_tokenized = self.abs_buckets_to_token(x_abs_bucketized, y_abs_bucketized)
        # assign a different range to relative positions
        rel_tokenized = self.rel_buckets_to_token(x_rel_bucketized, y_rel_bucketized)

        # print("self.relative_boundaries:", self.relative_boundaries)
        # print("x_abs_bucketized:", x_abs_bucketized)
        # print("y_abs_bucketized:", y_abs_bucketized)
        # print("x_rel_bucketized:", x_rel_bucketized)
        # print("y_rel_bucketized:", y_rel_bucketized)
        
        tokenized = torch.cat([abs_tokenized[..., [0]], rel_tokenized[..., 1:]], dim=-1)
        
        # put UNKNOWN token where data was unavailable (NaN)
        # exploit the rule that anything+nan=nan to reduce the coords dimension
        tokenized = torch.where(data.sum(dim=-1).isnan(), self.UNKNOWN, tokenized)
        
        return tokenized
    
    def set_device(self, device):
        self.x_boundaries = self.x_boundaries.to(device)
        self.y_boundaries = self.y_boundaries.to(device)
        self.relative_boundaries = self.relative_boundaries.to(device)
    
    # Thanks to Python typing, should work with ints and tensors alike, batched or unbatched
    # kinda like a y_buckets-base number
    def abs_buckets_to_token(self, x_abs_bucketized, y_abs_bucketized):
        return x_abs_bucketized * self.y_buckets + y_abs_bucketized + self.num_reserved_tokens
    
    # Thanks to Python typing, should work with ints and tensors alike, batched or unbatched
    def rel_buckets_to_token(self, x_rel_bucketized, y_rel_bucketized):
        return x_rel_bucketized * self.rel_buckets + y_rel_bucketized + self.x_buckets*self.y_buckets + self.num_reserved_tokens
    
    # again, kinda like a y_buckets-base number
    # Doesn't work with ints in this state, only tensors
    def __abs_token_to_buckets(self, token):
        token -= self.num_reserved_tokens
        x = token // self.y_buckets
        y = token % self.y_buckets
        return torch.stack((x,y), dim=-1)
    
    # Doesn't work with ints in this state, only tensors
    def __rel_token_to_buckets(self, token):
        token -= self.num_reserved_tokens + self.x_buckets*self.y_buckets
        x = token // self.rel_buckets
        y = token % self.rel_buckets
        return torch.stack((x,y), dim=-1)
    
    # keeping in mind that buckets 0 and len(boundaries) are "out-of-range",
    # each other bucket (i = 1, ..., len-1) is going to be the cell bounded by bounds[i-1] and bounds[i]
    def __abs_buckets_to_cellcenters(self, buckets):
        x_buckets = buckets[..., 0]
        y_buckets = buckets[..., 1]
        # setup to allow the "find centers" operation to work even in the presence of OOR buckets
        # (they will be filtered out later anyway)
        x_bounds_with_nan = torch.tensor(list(self.x_boundaries) + [torch.tensor(np.nan)], device=self.x_boundaries.device)
        y_bounds_with_nan = torch.tensor(list(self.y_boundaries) + [torch.tensor(np.nan)], device=self.y_boundaries.device)
                            # TODO cat?
        # for internal buckets, this does exactly as the comment above this function.
        # for bucket 0 (negative OOR) this gets bounds[0] and bounds[-1], thich is the final nan
        # for bucket len(boundaries) (positive OOR) this gets bounds[len-1] and bounds[len], which is again the final nan
        # so this should be fine. Then, filter out OOR results.
        x_centers = (x_bounds_with_nan[x_buckets-1] + x_bounds_with_nan[x_buckets]) / 2
        x_centers = torch.where(x_buckets == 0, -np.inf, x_centers)
        x_centers = torch.where(x_buckets == len(self.x_boundaries), np.inf, x_centers)  # TODO x_buckets rather than len(...)?
        y_centers = (y_bounds_with_nan[y_buckets-1] + y_bounds_with_nan[y_buckets]) / 2
        y_centers = torch.where(y_buckets == 0, -np.inf, y_centers)
        y_centers = torch.where(y_buckets == len(self.y_boundaries), np.inf, y_centers)
        return torch.stack((x_centers, y_centers), dim=-1)
    
    # same concept as above, but easier, b/c x and y have the same boundaries for relative
    def __rel_buckets_to_cellcenters(self, buckets):
        bounds_with_nan = torch.tensor(list(self.relative_boundaries) + [torch.tensor(np.nan)], device=self.relative_boundaries.device)
                            # TODO cat?
        centers = (bounds_with_nan[buckets-1] + bounds_with_nan[buckets]) / 2
        centers = torch.where(buckets==0, -np.inf, centers)
        centers = torch.where(buckets==len(self.relative_boundaries), np.inf, centers)
        return centers
    
    def token_to_buckets(self, token):
        # if token == self.UNKNOWN:
        #     return None  # TODO meglio tensore nan?
        # part 0
        tmp_token = token - self.num_reserved_tokens
        # to avoid that the abs functions end up manipulating the much larger relative tokens
        # the exact placeholder value doesn't really matter,
        # since that part will be filtered out by the other wheres
        abs_token = torch.where(tmp_token < self.x_buckets*self.y_buckets, token, 0)
        rel_token = torch.where(tmp_token >= self.x_buckets*self.y_buckets, token, 0)
        # for compatibility w/ abs_buk, rel_buk, abs_cen, rel_cen
        tmp_token = einops.repeat(tmp_token, "... token -> ... token coords", coords=2)
        # part 1
        abs_buk = self.__abs_token_to_buckets(abs_token)
        rel_buk = self.__rel_token_to_buckets(rel_token)
        the_buckets = torch.where(tmp_token < self.x_buckets*self.y_buckets, abs_buk, rel_buk)
        # part 2
        abs_cen = self.__abs_buckets_to_cellcenters(abs_buk)
        rel_cen = self.__rel_buckets_to_cellcenters(rel_buk)
        the_centers = torch.where(tmp_token < self.x_buckets*self.y_buckets, abs_cen, rel_cen)
        # filter out reserved tokens
        the_buckets = torch.where(tmp_token < 0, np.nan, the_buckets)
        the_centers = torch.where(tmp_token < 0, np.nan, the_centers)
        return the_buckets, the_centers
        # TODO MAYBE aggiungere precisione? (utile per darla al resto della challenge, forse)
        #            potrei anche fare una funzione diversa per fare centers e precision
        #            anziché caricare questa
    
    # TODO def token_is_abs(token)



if __name__ == "__main__":
    h,m,l = 20,5,6
    t = Tokenizer(
        AbsolutePosConf(-4500.0, -3000.0, 4500.0, 3000.0, 50, 30),
        RelativePosConf(2000.0, 3000.0, 6000.0, h, m, l, 2 * (h + m + l + 1)),
        ["UNKNOWN"]
    )
    test_batch = torch.rand((10,4,2)) * 6000
    print(test_batch)
    token = t.tokenize(test_batch)
    print(token.shape)
    print(t.token_to_buckets(token)[1])
