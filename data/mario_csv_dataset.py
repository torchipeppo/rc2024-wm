import torch
from torch.utils.data import Dataset

from pathlib import Path
import numpy as np
import pandas as pd
import einops

# Questo rimane per un solo commit, giusto perché ho fatto prima un cambio piccolo poi uno strutturale, e poi lo levo subito
class MarioCSVDataset_OLD(Dataset):
    def __init__(self, csv_path, number_of_other_robots, time_span):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        # default to searching in the same directory as this Python file (so maybe I won't need path_constants this time)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent / csv_path
        
        self.data = pd.read_csv(csv_path)
        self.data = self.data.sort_values(by=["frame", "ego_id", "id"])
        
        # +1 b/c we'll also want to fetch a target sequence, which is shifted by 1 wrt input
        self.frames_to_fetch = time_span + 1
        
        max_frame = max(self.data.frame.unique())
        self.frame_ego_pairs = self.data.filter(["frame", "ego_id"], axis="columns").drop_duplicates()
        # TODO sarebbe ancora più figo se ad ogni ego_id si scalassero le frames_to_fetch separatamente,
        #      dato che gli ID non sono sempre gli stessi per l'intera partita
        self.frame_ego_pairs = self.frame_ego_pairs[self.frame_ego_pairs.frame <= max_frame - self.frames_to_fetch + 1]
        
        self.number_of_other_robots = number_of_other_robots
        self.time_span = time_span
    
    def __len__(self):
        return self.frame_ego_pairs.shape[0]
    
    def __getitem__(self, idx):
        fep = self.frame_ego_pairs.iloc[idx]
        frame_range = range(fep.frame, fep.frame+self.frames_to_fetch)
        relevant_data = self.data[(self.data.frame.isin(frame_range)) & (self.data.ego_id == fep.ego_id)]
        
        # find closest robots
        other_robots_in_first_frame = relevant_data[
            (relevant_data.frame == fep.frame) &
            # (relevant_data.ego_id == fep.ego_id) &  # redundant
            (relevant_data.id != fep.ego_id) &
            (relevant_data.klasse == "robot")
        ].filter(["id", "relative_pos_x", "relative_pos_y"], axis="columns")
        other_robots_in_first_frame["sqr_distance"] = other_robots_in_first_frame.relative_pos_x ** 2 + other_robots_in_first_frame.relative_pos_y ** 2
        # there's only gonna be like 13 rows, we can afford a sort
        closest_robots = other_robots_in_first_frame.sort_values(by="sqr_distance").iloc[:self.number_of_other_robots]
        
        # track their IDs across the time span
        # extract absolute position of ego robot and relative position of other robots and ball
        ego_pos = relevant_data[relevant_data.id == fep.ego_id].filter(["field_pos_x", "field_pos_y"]).to_numpy()
        others_pos_list = [relevant_data[relevant_data.id == the_id].filter(["relative_pos_x", "relative_pos_y"]).to_numpy() for the_id in closest_robots.id]
        ball_pos = relevant_data[relevant_data.klasse == "ball"].filter(["relative_pos_x", "relative_pos_y"]).to_numpy()
        
        to_tensorize = einops.rearrange([ego_pos, *others_pos_list, ball_pos], "object time coords -> time object coords")
        return torch.Tensor(to_tensorize[:-1]), torch.Tensor(to_tensorize[1:])


# finché sono "solo" 100 MB, si può anche far tutto in RAM...
class MarioCSVDataset(Dataset):
    def __init__(self, csv_path, number_of_other_robots, time_span):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        # default to searching in the same directory as this Python file (so maybe I won't need path_constants this time)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent / csv_path
        
        # +1 b/c we'll also want to fetch a target sequence, which is shifted by 1 wrt input
        self.frames_to_fetch = time_span + 1
        self.data_by_egoid = dict()
        frame_ego_pairs_list = []
        
        for f in csv_path.iterdir():
            assert f.suffix == ".csv"
            egoid = int(f.stem)
            data = pd.read_csv(f).sort_values(by=["frame", "ego_id", "id"])
            max_frame = max(data.frame.unique())
            partial_frame_ego_pairs = data.filter(["frame", "ego_id"], axis="columns").drop_duplicates()
            partial_frame_ego_pairs = partial_frame_ego_pairs[partial_frame_ego_pairs.frame <= max_frame - self.frames_to_fetch + 1]
            
            self.data_by_egoid[egoid] = data
            frame_ego_pairs_list.append(partial_frame_ego_pairs)
        
        self.frame_ego_pairs = pd.concat(frame_ego_pairs_list).sort_values(by=["frame", "ego_id"])
        self.number_of_other_robots = number_of_other_robots
        self.time_span = time_span
    
    def __len__(self):
        return self.frame_ego_pairs.shape[0]
    
    def __getitem__(self, idx):
        fep = self.frame_ego_pairs.iloc[idx]
        frame_range = range(fep.frame, fep.frame+self.frames_to_fetch)
        relevant_data = self.data_by_egoid[fep.ego_id]
        relevant_data = relevant_data[relevant_data.frame.isin(frame_range)]
        
        # find closest robots
        other_robots_in_first_frame = relevant_data[
            (relevant_data.frame == fep.frame) &
            # (relevant_data.ego_id == fep.ego_id) &  # redundant
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
            ["field_pos_x", "field_pos_y"]
        )
        others_pos_list = [add_missing_frames_then_filter(
            relevant_data[relevant_data.id == the_id],
            frame_range,
            ["relative_pos_x", "relative_pos_y"]
        ) for the_id in closest_robots.id]
        ball_pos = add_missing_frames_then_filter(
            relevant_data[relevant_data.klasse == "ball"],
            frame_range,
            ["relative_pos_x", "relative_pos_y"]
        )
        
        to_tensorize = einops.rearrange([ego_pos, *others_pos_list, ball_pos], "object time coords -> time object coords")
        return torch.Tensor(to_tensorize[:-1]), torch.Tensor(to_tensorize[1:])




def add_missing_frames_then_filter(data, frame_range, columns_to_filter):
    for frame in frame_range:
        if data[data.frame == frame].empty:
            data = pd.concat([
                data,
                pd.DataFrame({"frame": frame, "field_pos_x": [np.nan], "field_pos_y": [np.nan], "relative_pos_x": [np.nan], "relative_pos_y": [np.nan]})
            ], ignore_index=True)
    return data.filter(columns_to_filter).to_numpy()


if __name__=="__main__":
    data = MarioCSVDataset("v1_processed_by_egoid", 2, 5)[0]
    
    print(data)
    # print(einops.rearrange(data, "... (coords two) -> ... coords two", two=2))
