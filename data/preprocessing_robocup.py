from pathlib import Path
import json
from enum import Enum
import math
import struct
import pandas as pd
import tqdm
import preprocessing_lib_obstacles as lib_obstacles

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

class DataEntryIndex(Enum):
    Version = 2-2
    PlayerNum = 3-2
    TeamNum = 4-2
    Fallen = 5-2
    PosX = 6-2
    PosY = 7-2
    Theta = 8-2
    BallAge = 9-2
    BallPosX = 10-2
    BallPosY = 11-2
    NumDataBytes = 12-2
    PlayerRole = 13-2
    CurrentObstacleSize = 14-2
    ObstacleTypes = 15-2
    ObstacleCenterX = 35-2
    ObstacleCenterY = 55-2
    ObstacleLastSeen = 75-2
    MessageBudget = 95-2
    SecsRemaining = 96-2
    ArmContact = 97-2
    ArmContactPos = 99-2
    ArmTimeOfLastContact = 101-2
    Suca2 = 103-2
    TeamBall = 104-2
    TeamBallVel = 106-2
    Suca3 = 108-2

# main

FILES = sorted((DATA_DIR/"ROBOCUP_TEST_DATA").rglob("*.jsonl"))
PROCESSES = 24

BALL_ID = 0
EGO_ID = 1  # this can be constant since we always have only the same controlled robot in these logs
FIRST_OBSTACLE_ID = 2

BALLAGE_THRESHOLD = 3  # ball age is in seconds...
OBSTACLEAGE_THRESHOLD = 4000  # ...but obstacle timestamps are in milliseconds
                            # keeping this more lenient than I'd like
                            # b/c I'm actually reading floats from the logs,
                            # due to an external oversight

def get_obstacle_list(record):
    obstacles_list = []
    obstacles_len = record[DataEntryIndex.CurrentObstacleSize.value]
    ego_x = record[DataEntryIndex.PosX.value]
    ego_y = record[DataEntryIndex.PosY.value]
    for i in range(obstacles_len):
        # see the ball part for all the comments
        x_absolute = record[DataEntryIndex.ObstacleCenterX.value + i]
        y_absolute = record[DataEntryIndex.ObstacleCenterY.value + i]
        x_relative = x_absolute - ego_x
        y_relative = y_absolute - ego_y
        # oh, and due to an external oversight, the JSONL logs actually store
        # the timestamp as though its byte representation was interpreted as a float,
        # not an unsigned int, so I need extra storage to store the "correct" timestamp.
        last_seen = record[DataEntryIndex.ObstacleLastSeen.value + i]
        # still putting an "if" here for future-proofing,
        # in case I use this again and the logs are fixed by then
        if type(last_seen) == float:
            # interpret last_seen's bytes as the unsigned int they should be
            last_seen = struct.unpack("<I", struct.pack("<f", last_seen))[0]
        
        obstacles_list.append(lib_obstacles.Obstacle(
            the_type=record[DataEntryIndex.ObstacleTypes.value + i],
            x_absolute=x_absolute,
            y_absolute=y_absolute,
            x_relative=x_relative,
            y_relative=y_relative,
            last_seen=last_seen,
        ))
    return obstacles_list

def handle_line(record, frame, tracker):
    partial_processed_list = []

    # ego row
    # (robot estimates its own position in absolute coordinates)
    ego_x = record[DataEntryIndex.PosX.value]
    ego_y = record[DataEntryIndex.PosY.value]
    partial_processed_list.append(dict(
        frame=frame,
        ego_id=EGO_ID,
        id=EGO_ID,
        klasse="robot",
        field_pos_x=ego_x,
        field_pos_y=ego_y,
        relative_pos_x=0.0,
        relative_pos_y=0.0,
        dist_to_ego=0.0,
    ))

    # ball row
    # (for our Shared Autonomy Challenge purposes, it was sent/logged as absolute coordinates)
    ball_age = record[DataEntryIndex.BallAge.value]
    if (0 < ball_age) and (ball_age <= BALLAGE_THRESHOLD):
        ball_x = record[DataEntryIndex.BallPosX.value]
        ball_y = record[DataEntryIndex.BallPosY.value]

        # then, in order to truly align with the MARIO data,
        # we want the relative ball position to ONLY be relative to the robot's position
        # (b/c MARIO can't get the orientation from the video)
        ball_x_relative = ball_x - ego_x
        ball_y_relative = ball_y - ego_y
        dist_to_ego = math.sqrt(ball_x_relative**2 + ball_y_relative**2)

        partial_processed_list.append(dict(
            frame=frame,
            ego_id=EGO_ID,
            id=0,
            klasse="ball",
            field_pos_x=ball_x,
            field_pos_y=ball_y,
            relative_pos_x=ball_x_relative,
            relative_pos_y=ball_y_relative,
            dist_to_ego=dist_to_ego,
        ))
    
    # now for everyone else
    # (for our Shared Autonomy Challenge purposes, it was sent/logged as absolute coordinates)
    # (get_obstacle_list already does all the things we'd do for the ball)
    obstacles = get_obstacle_list(record)
    filtered_obstacles_idxs = lib_obstacles.get_filtered_obstacles(obstacles, OBSTACLEAGE_THRESHOLD)

    # now for the actual stuff to do
    if len(filtered_obstacles_idxs) > 0:
        tracked_obstacles = tracker.track(obstacles)
        
        for obs in tracked_obstacles:
            dist_to_ego = math.sqrt(obs.x_relative**2 + obs.y_relative**2)
            partial_processed_list.append(dict(
                frame=frame,
                ego_id=EGO_ID,
                id=obs.the_id,
                klasse="robot",
                field_pos_x=obs.x_absolute,
                field_pos_y=obs.y_absolute,
                relative_pos_x=obs.x_relative,
                relative_pos_y=obs.y_relative,
                dist_to_ego=dist_to_ego,
            ))

    return pd.DataFrame(partial_processed_list)






def do_stuff(fname):
    print(fname)
    with open(fname, "r") as f:
        json_lines = f.readlines()
    
    tracker = lib_obstacles.HungarianTracker(FIRST_OBSTACLE_ID)
    
    processed_list = []
    
    for i in tqdm.tqdm(range(len(json_lines))):
        record = json.loads(json_lines[i])
        processed_list.append(handle_line(record, i, tracker))
    
    if len(processed_list) > 0:
        processed = pd.concat(processed_list)
        processed.to_csv(DATA_DIR / f"processed_real_test_set/{file_id_no}.csv", index=False)



file_id_no = 0

for fname in FILES:
    print(f"Files completed: {file_id_no} / {len(FILES)}")
    do_stuff(fname)
    file_id_no += 1
