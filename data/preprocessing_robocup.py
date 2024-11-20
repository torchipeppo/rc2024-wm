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
    Suca = 0
    Header = 1
    Version = 2
    PlayerNum = 3
    TeamNum = 4
    Fallen = 5
    PosX = 6
    PosY = 7
    Theta = 8
    BallAge = 9
    BallPosX = 10
    BallPosY = 11
    NumDataBytes = 12
    PlayerRole = 13
    CurrentObstacleSize = 14
    ObstacleTypes = 15
    ObstacleCenterX = 35
    ObstacleCenterY = 55
    ObstacleLastSeen = 75
    MessageBudget = 95
    SecsRemaining = 96
    ArmContact = 97
    ArmContactPos = 99
    ArmTimeOfLastContact = 101
    Suca2 = 103
    TeamBall = 104
    TeamBallVel = 106
    Suca3 = 108

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

# this formula is the very same the B-Human framework uses for Pose2f::operator*
# (which itself is most likely taken from the eigen library).
def rel2glob_with_theta(x_relative, y_relative, ego_x, ego_y, ego_theta):
    c = math.cos(ego_theta)
    s = math.sin(ego_theta)
    x = x_relative * c - y_relative * s + ego_x
    y = x_relative * s + y_relative * c + ego_y
    return x, y

def get_obstacle_list(record):
    obstacles_list = []
    obstacles_len = record[DataEntryIndex.CurrentObstacleSize.value]
    ego_x = record[DataEntryIndex.PosX.value]
    ego_y = record[DataEntryIndex.PosY.value]
    ego_theta = record[DataEntryIndex.Theta.value]
    for i in range(obstacles_len):
        # see the ball part for all the comments
        x_original = record[DataEntryIndex.ObstacleCenterX.value + i]
        y_original = record[DataEntryIndex.ObstacleCenterY.value + i]
        x_absolute, y_absolute = rel2glob_with_theta(x_original, y_original, ego_x, ego_y, ego_theta)
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
            x_original=x_original,
            y_original=y_original,
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
    # (robot estimates ball in local coordinates, relative to both position and orientation)
    ball_age = record[DataEntryIndex.BallAge.value]
    if (0 < ball_age) and (ball_age <= BALLAGE_THRESHOLD):
        ball_x_relative = record[DataEntryIndex.BallPosX.value]
        ball_y_relative = record[DataEntryIndex.BallPosY.value]
        dist_to_ego = math.sqrt(ball_x_relative**2 + ball_y_relative**2)

        # BallModel is relative to their own position and orientation
        # so, converting this local position to field coordinates requires translation and rotation
        ego_theta = record[DataEntryIndex.Theta.value]
        ball_x, ball_y = rel2glob_with_theta(ball_x_relative, ball_y_relative, ego_x, ego_y, ego_theta)

        # but it's not over yet: in order to truly align with the MARIO data,
        # we want the relative ball position to ONLY be relative to the robot's position
        # (b/c MARIO can't get the orientation from the video)
        ball_x_relative = ball_x - ego_x
        ball_y_relative = ball_y - ego_y

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
    # (there we rely on data from the robot's ObstacleModel, which works in local coordinates)
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





file_id_no = 0
print(f"Files completed: {file_id_no} / {len(FILES)}")

def do_stuff(fname):
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

for fname in FILES:
    do_stuff(fname)