import numpy as np
import pandas as pd
import ast
import tqdm
import multiprocessing

def select(data, col_name, col_value):
    return data.loc[data[col_name] == col_value]

def get_frame(data, frame_idx):
    # return data.loc[data.frame == frame_idx]
    return select(data, "frame", frame_idx)

# # add either a bunch of already-made DataFrames, or a single row via kwargs
# # NO SIDE EFFECT ON DATA
# def appended(data, *new_dfs, **new_row_kwargs):
#     if len(new_dfs) > 0:
#         return pd.concat([data] + new_dfs)
#     else:
#         return pd.concat([data, pd.DataFrame(new_row_kwargs)])

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
        # breakpoint()
        relative_to_new = interpolate(relative_to, IN_FIELD_MIN, IN_FIELD_MAX, OUT_FIELD_MIN, OUT_FIELD_MAX)

        field_pos_new = field_pos_new - relative_to_new
    
    # breakpoint()
    return field_pos_new

def do_frame_ego_pair(frame_idx, frame_data, ego_id):
    processed_list = []
    ego_row = select(frame_data, "id", ego_id)
    ego_pos = process_field_pos(ego_row.field_pos)
    ego_pos_relative = process_field_pos(ego_row.field_pos, relative_to=ego_row.field_pos)
    processed_list.append(pd.DataFrame(dict(
        frame=frame_idx,
        ego_id=ego_id,
        id=ego_id,
        klasse=ego_row.klasse,
        field_pos_x=ego_pos[0],
        field_pos_y=ego_pos[1],
        relative_pos_x=ego_pos_relative[0],
        relative_pos_y=ego_pos_relative[1],
    ), columns=COLUMNS))
    
    for other_id in frame_data.id:
        other_row = select(frame_data, "id", other_id)
        if other_id != ego_id:
            other_pos = process_field_pos(other_row.field_pos)
            other_pos_relative = process_field_pos(other_row.field_pos, relative_to=ego_row.field_pos)
            processed_list.append(pd.DataFrame(dict(
                frame=frame_idx,
                ego_id=ego_id,
                id=other_id,
                klasse=other_row.klasse,
                field_pos_x=other_pos[0],
                field_pos_y=other_pos[1],
                relative_pos_x=other_pos_relative[0],
                relative_pos_y=other_pos_relative[1],
            ), columns=COLUMNS))
    
    # processed = pd.concat(processed_list)
    # return processed

    return processed_list

def tuple_string_to_numpy(s):
    t = ast.literal_eval(s)
    a = np.array(t)
    return a

# class TheThread(threading.Thread):
#     def __init__(self, data, frame_indices):
#         super().__init__()
#         self.data = data
#         self.frame_indices = frame_indices
#         self.done = False
    
#     def run(self):
#         self.processed_list = []
#         for frame_idx in tqdm.tqdm(self.frame_indices):
#             frame_data = get_frame(self.data, frame_idx)
#             for ego_id in frame_data.id:
#                 if (select(frame_data, "id", ego_id).klasse == "robot").item():
#                     self.processed_list.extend(do_frame_ego_pair(frame_idx, frame_data, ego_id))
#         self.done = True
    
#     def result(self):
#         if not self.done:
#             raise RuntimeError("Thread not done!")
#         return self.processed_list

def process_job(data, frame_indices, output_list):
    for frame_idx in tqdm.tqdm(frame_indices):
        frame_data = get_frame(data, frame_idx)
        for ego_id in frame_data.id:
            if (select(frame_data, "id", ego_id).klasse == "robot").item():
                output_list.extend(do_frame_ego_pair(frame_idx, frame_data, ego_id))
    # "returns" to main process through side-effect on output_list

COLUMNS = ["frame", "ego_id", "id", "klasse", "field_pos_x", "field_pos_y", "relative_pos_x", "relative_pos_y"]

# def new_processed():
#     return pd.DataFrame(columns=COLUMNS)


# main

data = pd.read_csv(
    "v1--output.csv",
    names=["frame", "id", "klasse", "color", "image_bbox", "field_pos"],
    converters={"field_pos": tuple_string_to_numpy}
)

processed_list = []

THREADS = 8

frame_indices_by_thread = np.array_split(data.frame.unique(), THREADS)
manager = multiprocessing.Manager()
output_lists = [manager.list() for _ in range(THREADS)]

procs = [multiprocessing.Process(target=process_job, args=(data, frame_indices_by_thread[i], output_lists[i])) for i in range(THREADS)]

for i in range(len(procs)):
    procs[i].start()

for i in range(len(procs)):
    procs[i].join()
    # processed_list.extend(t.result())
    processed_list.extend(output_lists[i])

# for frame_idx in tqdm.tqdm(data.frame.unique()):
#     frame_data = get_frame(data, frame_idx)
#     for ego_id in frame_data.id:
#         if (select(frame_data, "id", ego_id).klasse == "robot").item():
#             processed_list.append(do_frame_ego_pair(frame_idx, frame_data, ego_id))

processed = pd.concat(processed_list)

processed.to_csv("v1--processed.csv")
processed.to_hdf("v1--processed.h5", key="data")
