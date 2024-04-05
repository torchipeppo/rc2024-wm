import torch
from torch.utils.data import Dataset

from pathlib import Path
import pandas as pd
import einops

# finché sono "solo" 100 MB, si può anche far tutto in RAM...
class MarioCSVDataset(Dataset):
    def __init__(self, csv_path, number_of_other_robots, time_span):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        
        self.data = pd.read_csv("v1--processed.csv")
        self.data = self.data.sort_values(by=["frame", "ego_id", "id"])
        
        max_frame = max(self.data.frame.unique())
        self.frame_ego_pairs = self.data.filter(["frame", "ego_id"], axis="columns").drop_duplicates()
        # TODO sarebbe ancora più figo se ad ogni ego_id si scalasse il time_span separatamente,
        #      dato che gli ID non sono sempre gli stessi per l'intera partita
        self.frame_ego_pairs = self.frame_ego_pairs[self.frame_ego_pairs.frame <= max_frame - time_span + 1]
        
        self.number_of_other_robots = number_of_other_robots
        self.time_span = time_span
    
    def __len__(self):
        return self.frame_ego_pairs[0]
    
    def __getitem__(self, idx):
        fep = self.frame_ego_pairs.iloc[idx]
        relevant_data = self.data[(self.data.frame.isin(range(fep.frame, fep.frame+self.time_span))) & (self.data.ego_id == fep.ego_id)]
        
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
        to_tensorize = einops.rearrange([ego_pos, *others_pos_list, ball_pos], "object time coords -> time (object coords)")
        return torch.Tensor(to_tensorize)


if __name__=="__main__":
    print(MarioCSVDataset("v1--processed.csv", 2, 5)[0])
