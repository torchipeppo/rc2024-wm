import torch
import numpy as np
import einops

from dataclasses import dataclass

@dataclass
class MetricOutput:
    abs: torch.tensor
    rel: torch.tensor
    glob: torch.tensor

# euclidean distance of field positions
def field_distance(pred_field_pos, true_field_pos, abs_token_mask):
    x = (true_field_pos - pred_field_pos) ** 2
    x = einops.reduce(x, "batch object coords -> batch object", "sum")
    x = torch.sqrt(x)
    abs_field_distance = torch.nanmean(torch.where(abs_token_mask, x, np.nan))
    rel_field_distance = torch.nanmean(torch.where(torch.logical_not(abs_token_mask), x, np.nan))
    glob_field_distance = torch.nanmean(x)
    # if (glob_field_distance.isinf()):
    #     breakpoint()
    return MetricOutput(abs_field_distance, rel_field_distance, glob_field_distance)

# manhattan distance in grid space
def grid_distance(pred_grid_pos, targ_grid_pos, abs_token_mask):
    x = torch.abs(targ_grid_pos - pred_grid_pos)
    x = einops.reduce(x, "batch object coords -> batch object", "sum")
    x = x.to(float)
    abs_grid_distance = torch.nanmean(torch.where(abs_token_mask, x, np.nan))
    rel_grid_distance = torch.nanmean(torch.where(torch.logical_not(abs_token_mask), x, np.nan))
    glob_grid_distance = torch.nanmean(x)
    return MetricOutput(abs_grid_distance, rel_grid_distance, glob_grid_distance)
