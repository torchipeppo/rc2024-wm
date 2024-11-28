from pathlib import Path
import json
from enum import Enum
import math
import struct
import pandas as pd
import tqdm

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

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
for f in FILES:
    # this one file is actually pretty bad and doesn't correspond to a full game,
    # I don't want it in this calculation
    if "rUNSWift-test-1" in f.name:
        FILES.remove(f)


all_occurrs_of_each_secsrem = []

for fname in FILES:
    with open(fname, "r") as f:
        json_lines = f.readlines()
    
    occurrences_of_each_secsremaining = [0 for _ in range(91)]
    
    for line in json_lines:
        record = json.loads(line)
        secs_remaining = record[DataEntryIndex.SecsRemaining.value]
        if secs_remaining <= 90:
            occurrences_of_each_secsremaining[secs_remaining] += 1

    print(fname)
    for i in range(len(occurrences_of_each_secsremaining)):
        print(f"{i} secs for {occurrences_of_each_secsremaining[i]} frames")
    
    all_occurrs_of_each_secsrem.append(occurrences_of_each_secsremaining)


# let's do a little statistical analysis
data = []
for occurrences_of_each_secsremaining in all_occurrs_of_each_secsrem:
    for secsremaining in range(len(occurrences_of_each_secsremaining)):
        # playing signal is delayed by 15 seconds, so we get told 90 for the first 15 seconds,
        # making the first 15 secods unreliable for this purpose
        if 0 < secsremaining and secsremaining <= 75:
            data.append(occurrences_of_each_secsremaining[secsremaining])
mean = sum(data) / len(data)
mode = max(set(data), key=data.count)  # https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
median = sorted(data)[len(data)//2]
print(f"mean: {mean}")
print(f"mode: {mode}")
print(f"median: {median}")
print("mode uniqueness check: does the highest number in the following list appear only once?")
print(sorted([data.count(i) for i in set(data)]))
print(f"{sorted(set(data), key=data.count)}")
