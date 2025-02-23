import pandas as pd
import multiprocessing
import math
from pathlib import Path
import tqdm

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def tp2id(team, player):
    return team*100 + player

def sr2frame(secsremaining):
    return (600-secsremaining) * FPS

def select(data, col_name, col_value):
    return data.loc[data[col_name] == col_value]

def handle_egorow(egorow, data, tp_pairs):
    relevant_data = data.loc[data.secsremaining <= egorow.secsremaining]
    ego_id = tp2id(egorow.team, egorow.player)
    frame = sr2frame(egorow.secsremaining)

    partial_processed_list = []

    # one data row for each robot
    # (robot estimates its own position in absolute coordinates)
    ego_x = egorow.x
    ego_y = egorow.y
    partial_processed_list.append(dict(
        frame=frame,
        ego_id=ego_id,
        id=ego_id,
        klasse="robot",
        field_pos_x=ego_x,
        field_pos_y=ego_y,
        relative_pos_x=0.0,
        relative_pos_y=0.0,
        dist_to_ego=0.0,
    ))

    # one data row for the ball
    # (robot estimates ball in local coordinates, relative to both position and orientation)
    if (0 < egorow.ballage) and (egorow.ballage <= BALLAGE_THRESHOLD):
        ball_x_relative = egorow.ballx
        ball_y_relative = egorow.bally
        dist_to_ego = math.sqrt(ball_x_relative**2 + ball_y_relative**2)

        # robots tell the GC the ball position relative to their own position and orientation
        # so, converting this local position to field coordinates requires translation and rotation
        # this formula is the very same the B-Human framework uses for Pose2f::operator*
        # (which itself is most likely taken from the eigen library).
        c = math.cos(egorow.theta)
        s = math.sin(egorow.theta)
        ball_x = ball_x_relative * c - ball_y_relative * s + ego_x
        ball_y = ball_x_relative * s + ball_y_relative * c + ego_y

        # but it's not over yet: in order to truly align with the MARIO data,
        # we want the relative ball position to ONLY be relative to the robot's position
        # (b/c MARIO can't get the orientation from the video)
        ball_x_relative = ball_x - ego_x
        ball_y_relative = ball_y - ego_y

        partial_processed_list.append(dict(
            frame=frame,
            ego_id=ego_id,
            id=0,
            klasse="ball",
            field_pos_x=ball_x,
            field_pos_y=ball_y,
            relative_pos_x=ball_x_relative,
            relative_pos_y=ball_y_relative,
            dist_to_ego=dist_to_ego,
        ))
    
    # now for everyone else
    # (we take other robots' position from their own estimates, so they're in global coords)
    for tp in tp_pairs:
        if tp2id(*tp) != ego_id:
            # selector = relevant_data.team == tp[0] and relevant_data.player == tp[1]
            other_data = relevant_data[relevant_data.team == tp[0]]
            other_data = other_data[other_data.player == tp[1]]
            if other_data.shape[0] > 0:  # not all tp pairs might exist
                other_data.sort_values(by="secsremaining", ascending=False, inplace=True)
                other_row = other_data.iloc[0]
                other_id = tp2id(other_row.team.item(), other_row.player.item())
                other_x = other_row.x.item()
                other_y = other_row.y.item()
                other_x_relative = other_x - ego_x
                other_y_relative = other_y - ego_y
                dist_to_ego = math.sqrt(other_x_relative**2 + other_y_relative**2)
                partial_processed_list.append(dict(
                    frame=frame,
                    ego_id=ego_id,
                    id=other_id,
                    klasse="robot",
                    field_pos_x=other_x,
                    field_pos_y=other_y,
                    relative_pos_x=other_x_relative,
                    relative_pos_y=other_y_relative,
                    dist_to_ego=dist_to_ego,
                ))

    return pd.DataFrame(partial_processed_list)




# main

FILES = sorted((DATA_DIR/"GC_DATA").glob("log*ALL.csv"))
PROCESSES = 56

FPS = 30
BALLAGE_THRESHOLD = 3


file_id_no = 10000
print(f"Files completed: {file_id_no} / {len(FILES)}")


for fname in FILES:
    # would have liked to make this a function, but it appears that internal functions
    # are incompatible w/ multiprocessing due to pickling

    data = pd.read_csv(fname)
    # filter
    data = select(data, "playing", True)

    teams = data.team.unique()
    players = data.player.unique()
    tp_pairs = []
    for t in teams:
        for p in players:
            tp_pairs.append((t,p))

    def do_egoid_file(tp_pair):
        team, player = tp_pair
        ego_id = tp2id(team, player)
        processed_list = []
        for _, row in data.iterrows():
            if (row.goodego == True) and (row.team == team) and (row.player == player):
                processed_list.append(handle_egorow(row, data, tp_pairs))
        if processed_list:
            processed = pd.concat(processed_list)
            processed.to_csv(DATA_DIR / f"processed_by_gameegoid/{file_id_no}-{ego_id}.csv", index=False)

    pool = multiprocessing.Pool(PROCESSES)
    progressbar = tqdm.tqdm(total=len(tp_pairs))
    for _ in pool.imap_unordered(do_egoid_file, tp_pairs, chunksize=1):
        progressbar.update()
    pool.close()
    pool.join()
    progressbar.close()

    file_id_no += 1
    print(f"Files completed: {file_id_no} / {len(FILES)}")
