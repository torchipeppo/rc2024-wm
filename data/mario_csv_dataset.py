import torch
from torch.utils.data import Dataset

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import einops

FPS = 30

# a defaultdict that supplies the key to the factory
class defaultdict_ext(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        return self.default_factory(key)

# finché sono "solo" 100 MB, si può anche far tutto in RAM...
class MarioCSVDataset(Dataset):
    def __init__(self, csv_path, number_of_other_robots, time_span, min_time_span, time_bw_frames):
        assert min_time_span > 0
        assert time_span >= min_time_span
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        # default to searching in the same directory as this Python file (so maybe I won't need path_constants this time)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent / csv_path
        
        # +1 b/c we'll also want to fetch a target sequence, which is shifted by 1 wrt input
        self.frames_to_fetch = time_span + 1
        self.data_by_egoid = dict()
        self.maxframe_by_egoid = dict()
        frame_ego_pairs_list = []
        
        for f in csv_path.iterdir():
            assert f.suffix == ".csv"
            egoid = int(f.stem)
            data = pd.read_csv(f).sort_values(by=["frame", "ego_id", "id"])
            max_frame = max(data.frame.unique())
            partial_frame_ego_pairs = data.filter(["frame", "ego_id"], axis="columns").drop_duplicates()
            partial_frame_ego_pairs = partial_frame_ego_pairs[partial_frame_ego_pairs.frame <= max_frame - (min_time_span-1)*time_bw_frames]
            
            self.data_by_egoid[egoid] = data
            self.maxframe_by_egoid[egoid] = max_frame
            frame_ego_pairs_list.append(partial_frame_ego_pairs)
        
        self.frame_ego_pairs = pd.concat(frame_ego_pairs_list).sort_values(by=["frame", "ego_id"])
        self.number_of_other_robots = number_of_other_robots
        self.time_span = time_span
        self.time_bw_frames = time_bw_frames
        self.next_frame_tolerance = time_bw_frames // 10
    
    def __len__(self):
        return self.frame_ego_pairs.shape[0]
    
    def __getitem__(self, idx):
        fep = self.frame_ego_pairs.iloc[idx]
        frame_range = [fep.frame + self.time_bw_frames * i for i in range(self.frames_to_fetch)]
        actually_found_frames = defaultdict_ext(lambda k: k)
        relevant_data = self.data_by_egoid[fep.ego_id]
        
        relevant_data_list = []
        for f in frame_range:
            try:
                df = find_closest_frame(relevant_data, f, self.next_frame_tolerance)
                relevant_data_list.append(df)
                actually_found_frames[f] = df.iloc[0].frame
            except ClosestFrameNotFoundWithinToleranceException:
                break
        relevant_data = pd.concat(relevant_data_list)
        
        # find closest robots
        other_robots_in_first_frame = relevant_data[
            (relevant_data.frame == fep.frame) &
            (relevant_data.id != fep.ego_id) &
            (relevant_data.klasse == "robot")
        ].filter(["id", "relative_pos_x", "relative_pos_y"], axis="columns")
        other_robots_in_first_frame["sqr_distance"] = other_robots_in_first_frame.relative_pos_x ** 2 + other_robots_in_first_frame.relative_pos_y ** 2
        # there's only gonna be like 13 rows, we can afford a sort
        closest_robots = other_robots_in_first_frame.sort_values(by="sqr_distance").iloc[:self.number_of_other_robots]
        
        # track their IDs across the time span
        # extract absolute position of ego robot and relative position of other robots and ball
        ego_pos = add_missing_frames_then_filter(
            relevant_data[relevant_data.id == fep.ego_id],
            frame_range,
            ["field_pos_x", "field_pos_y"],
            actually_found_frames=actually_found_frames
        )
        others_pos_list = [add_missing_frames_then_filter(
            relevant_data[relevant_data.id == the_id],
            frame_range,
            ["relative_pos_x", "relative_pos_y"],
            actually_found_frames=actually_found_frames
        ) for the_id in closest_robots.id]
        ball_pos = add_missing_frames_then_filter(
            relevant_data[relevant_data.klasse == "ball"],
            frame_range,
            ["relative_pos_x", "relative_pos_y"],
            actually_found_frames=actually_found_frames
        )
        
        to_tensorize = einops.rearrange([ego_pos, *others_pos_list, ball_pos], "object time coords -> time object coords")
        return torch.Tensor(to_tensorize[:-1]), torch.Tensor(to_tensorize[1:])

class ClosestFrameNotFoundWithinToleranceException(ValueError):
    pass

def find_closest_frame(data, required_frame, tolerance):
    tol_range = range(required_frame-tolerance, required_frame+tolerance+1)  # +1 b/c I like the symmetric range [rq-tol, rq+tol] w/ both ends included
    # candidate_frames = data[required_frame-tolerance <= data.frame <= required_frame+tolerance]
    candidate_data = data[data.frame.isin(tol_range)]
    if len(candidate_data) == 0:
        raise ClosestFrameNotFoundWithinToleranceException()  # return ERROR_VALUE  # make_nan_frame(required_frame)
    candidate_frames = pd.DataFrame(candidate_data.frame.unique(), columns=["frame"])  # dunno if there's a better way
    candidate_frames.loc[:, "dist_from_required"] = abs(candidate_frames.frame - required_frame)
    the_frame = candidate_frames.sort_values(by="dist_from_required").iloc[0].frame
    return candidate_data[candidate_data.frame == the_frame]

def add_missing_frames_then_filter(data, frame_range, columns_to_filter, *, actually_found_frames=None):
    if actually_found_frames is None:
        actually_found_frames = defaultdict_ext(lambda k: k)  # retain behavior from before actually_found_frames was added
    for frame in frame_range:
        if data[data.frame == actually_found_frames[frame]].empty:
            data = pd.concat([
                data,
                make_nan_frame(frame)
            ], ignore_index=True)
    return data.filter(columns_to_filter).to_numpy()

def make_nan_frame(frame_no):
    return pd.DataFrame({"frame": frame_no, "field_pos_x": [np.nan], "field_pos_y": [np.nan], "relative_pos_x": [np.nan], "relative_pos_y": [np.nan]})

def cut_at_first_occurrence_of(l, x):
    i=0
    for elem in l:
        if elem == x:
            return l[:i]  # i is excluded
        else:
            i+=1
    # if we get here there is no None
    return l

if __name__=="__main__":
    data = MarioCSVDataset(
        "v1_processed_by_egoid",
        number_of_other_robots=2,
        min_time_span=3,
        time_span=5,
        time_bw_frames=10*FPS)[3]
    # TODO NEXT: decidere per bene time_bw_frames in config
    
    print(data[0])
