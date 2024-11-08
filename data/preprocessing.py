import numpy as np
import pandas as pd
import ast
import tqdm
import multiprocessing
from pathlib import Path
import math

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def select(data, col_name, col_value):
    return data.loc[data[col_name] == col_value]

def get_frame(data, frame_idx):
    # return data.loc[data.frame == frame_idx]
    return select(data, "frame", frame_idx)

IN_FIELD_MIN = np.array((0,0))
IN_FIELD_MAX = np.array((800, 600))
OUT_FIELD_MIN = np.array((-4500, -3000))
OUT_FIELD_MAX = np.array((4500, 3000))

def interpolate(pos, in_min, in_max, out_min, out_max):
    pos_in_01 = (pos - in_min) / (in_max - in_min)
    return pos_in_01 * (out_max - out_min) + out_min

# map from [0,800]x[0,600] to [-4500,4500]x[-3000,3000]
def process_field_pos(field_pos, *, relative_to=None):    
    if isinstance(field_pos, pd.Series):
        field_pos = field_pos.iat[0]
    
    assert field_pos.shape == (2,), "One at a time pls"
    field_pos_new = interpolate(field_pos, IN_FIELD_MIN, IN_FIELD_MAX, OUT_FIELD_MIN, OUT_FIELD_MAX)
    
    if relative_to is not None:
        if isinstance(relative_to, pd.Series):
            relative_to = relative_to.iat[0]
        relative_to_new = interpolate(relative_to, IN_FIELD_MIN, IN_FIELD_MAX, OUT_FIELD_MIN, OUT_FIELD_MAX)
        
        field_pos_new = field_pos_new - relative_to_new
    
    return field_pos_new

def do_frame_ego_pair(frame_idx, frame_data, ego_id):
    processed_list = []
    ego_row = select(frame_data, "id", ego_id)
    
    # aggressive strategy: if more than one ball is reported in a frame, don't trust anything
    trust_ball = (len(select(frame_data, "klasse", "ball")) <= 1)
    
    for other_id in frame_data.id:
        other_row = select(frame_data, "id", other_id)
        if trust_ball or other_row.klasse.item() != "ball":
            # (this data comes from video analysis, so all positions are in global coordinates)
            other_pos = process_field_pos(other_row.field_pos)
            other_pos_relative = process_field_pos(other_row.field_pos, relative_to=ego_row.field_pos)
            dist_to_ego = math.sqrt(other_pos_relative[0]**2 + other_pos_relative[1]**2)
            processed_list.append(pd.DataFrame(dict(
                frame=frame_idx,
                ego_id=ego_id,
                id=other_id,
                klasse=other_row.klasse,
                field_pos_x=other_pos[0],
                field_pos_y=other_pos[1],
                relative_pos_x=other_pos_relative[0],
                relative_pos_y=other_pos_relative[1],
                dist_to_ego=dist_to_ego,
            ), columns=COLUMNS))
    
    return processed_list

def tuple_string_to_numpy(s):
    t = ast.literal_eval(s)
    a = np.array(t)
    return a

COLUMNS = ["frame", "ego_id", "id", "klasse", "field_pos_x", "field_pos_y", "relative_pos_x", "relative_pos_y", "dist_to_ego"]

# main

# leaving the construction of the FILES list like this in order not to change
# the order of the files, and thus their ID numbers, for this version of the dataset.
# But if I ever make an expanded version, we can definitely switch to a much simpler:
# FILES = sorted((DATA_DIR/"MARIO_DATA").glob("*.csv"))
FILES = [
    DATA_DIR/"MARIO_DATA" / "unknown-game.csv",
    DATA_DIR/"MARIO_DATA" / "German Open 2024 - SPL - Field A - Final Day__998.0.csv",
    DATA_DIR/"MARIO_DATA" / "German Open 2024 - SPL - Field A - Final Day__2907.0.csv",
]
FILES.extend(sorted((DATA_DIR/"MARIO_DATA").glob("RoboCup 2023 - SPL - Field *.csv")))
PROCESSES = 56


if __name__ == "__main__":

    PROCESSED_DIR = DATA_DIR / "processed_by_gameegoid"
    PROCESSED_DIR.mkdir(exist_ok=True)

    file_id_no = 0
    print(f"Files completed: {file_id_no} / {len(FILES)}")


    for fname in FILES:
        # would have liked to make this a function, but it appears that internal functions
        # are incompatible w/ multiptocessing due to pickling

        data = pd.read_csv(
            fname,
            names=["frame", "id", "klasse", "color", "image_bbox", "field_pos"],
            converters={"field_pos": tuple_string_to_numpy}
        )

        frames = data.frame.unique()
        ids = data.id.unique()

        def do_egoid_file(ego_id):
            processed_list = []
            for frame in frames:
                frame_data = get_frame(data, frame)
                if (ego_id in frame_data.id.values) and (select(frame_data, "id", ego_id).klasse == "robot").item():
                    processed_list.extend(do_frame_ego_pair(frame, frame_data, ego_id))
            if processed_list:
                processed = pd.concat(processed_list)
                processed.to_csv(PROCESSED_DIR / f"{file_id_no}-{ego_id}.csv", index=False)

        pool = multiprocessing.Pool(PROCESSES)
        progressbar = tqdm.tqdm(total=len(ids))
        for _ in pool.imap_unordered(do_egoid_file, ids, chunksize=10):
            progressbar.update()
        pool.close()
        pool.join()
        progressbar.close()

        file_id_no += 1
        print(f"Files completed: {file_id_no} / {len(FILES)}")
