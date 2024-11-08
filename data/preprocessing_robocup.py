from pathlib import Path
import json
from enum import Enum
import math
import struct

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
class ObstacleType(Enum):
    Nothing = -1
    Goalpost = 0
    Unknown = 1
    SomeRobot = 2
    Opponent = 3
    Teammate = 4
    FallenSomeRobot = 5
    FallenOpponent = 6
    FallenTeammate = 7
ROBOT_TYPES = {
    ObstacleType.SomeRobot.value,
    ObstacleType.Opponent.value,
    ObstacleType.Teammate.value,
    ObstacleType.FallenSomeRobot.value,
    ObstacleType.FallenOpponent.value,
    ObstacleType.FallenTeammate.value,
}

# main

FILES = sorted((DATA_DIR/"ROBOCUP_TEST_DATA").rglob("*.jsonl"))
PROCESSES = 24

EGO_ID = 1  # this can be constant since we always have only the same controlled robot in these logs
BALLAGE_THRESHOLD = 3  # ball age is in seconds...
OBSTACLEAGE_THRESHOLD = 4000  # ...but obstacle timestamps are in milliseconds
                            # keeping this more lenient than I'd like
                            # b/c I'm actually reading floats from the logs,
                            # due to an external oversight

file_id_no = 0
print(f"Files completed: {file_id_no} / {len(FILES)}")

def handle_line(record, frame):
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
        # this formula is the very same the B-Human framework uses for Pose2f::operator*
        # (which itself is most likely taken from the eigen library).
        theta = record[DataEntryIndex.Theta.value]
        c = math.cos(theta)
        s = math.sin(theta)
        ball_x = ball_x_relative * c - ball_y_relative * s + ego_x
        ball_y = ball_x_relative * s + ball_y_relative * c + ego_y

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
    obstacles_len = record[DataEntryIndex.CurrentObstacleSize.value]
    # filter: only consider robots (not goalposts)
    robot_obstacle_idxs = []
    for i in range(obstacles_len):
        obstacle_type = record[DataEntryIndex.ObstacleTypes.value + i]
        if obstacle_type in ROBOT_TYPES:
            robot_obstacle_idxs.append(i)

    if len(robot_obstacle_idxs) > 0:
        # filter: only consider recent ddetections
        # but...
        # turns out the robot was instructed to send the absolute timestamp
        # of last detection of each obstacle, but no way to get the "obstacle age"
        # oh well, I'll do the best I can and exclude obstacles that are much older
        # than the most recent one.
        most_recent_obstacle_seen = -1
        # oh, and due to an external oversight, the JSONL logs actually store
        # the timestamp as though its byte representation was interpreted as a float,
        # not an unsigned int, so I need extra storage to store the "correct" timestamp.
        obstacles_last_seen = dict()
        for i in robot_obstacle_idxs:
            time_seen = record[DataEntryIndex.ObstacleLastSeen.value]
            # putting an "if" is for future-proofing,
            # in case I use this again and the logs are fixed by then
            if type(time_seen) == float:
                # interpret time_seen's bytes as the unsigned int they should be
                time_seen = struct.unpack("<I", struct.pack("<f", time_seen))[0]
            obstacles_last_seen[i] = time_seen
            most_recent_obstacle_seen = max(most_recent_obstacle_seen, time_seen)
        
        # now for the actual stuff to do (while filtering for recency)
        for i in robot_obstacle_idxs:
            obstacle_age_approx = most_recent_obstacle_seen - obstacles_last_seen[i]
            if obstacle_age_approx < OBSTACLEAGE_THRESHOLD:
                # TODO obstacle tracking (yeeeeee)





def do_stuff(fname):
    with open(fname, "r") as f:
        json_lines = f.readlines()
    
    processed_list = []
    
    for i in len(json_lines):
        record = json.loads(json_lines[i])
        processed_list.extend(handle_line(record, i))
