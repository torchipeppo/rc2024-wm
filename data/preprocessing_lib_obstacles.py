from enum import Enum
from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np

from preprocessing_lib_hungarian import hungarian

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
TEAMMATE_TYPES = {
    ObstacleType.Teammate,
    ObstacleType.FallenTeammate,
}
OPPONENT_TYPES = {
    ObstacleType.Opponent,
    ObstacleType.FallenOpponent,
}

UNASSIGNED_ID = -14383421  # truth had gone, truth had gone, and truth had gone
TRACKING_EXCLUSION_THRESHOLD = 500.0  # TODO this value is quite random

# much greater than any possible distance achievable on the 9000x6000 field,
# without risking with infinity nonsense.
# If legit obstacle pairs end up equating or exceeding this value,
# then I have bigger problems than the tracking algorithm failing,
# because either my data contains invalid positions or I'm preprocessing them wrong!
INVALID_WEIGHT = 999999.999


@dataclass
class Obstacle:
    the_type: ObstacleType
    x_absolute: float
    y_absolute: float
    x_relative: float
    y_relative: float
    last_seen: int
    the_id: int = UNASSIGNED_ID

    def is_perfect_match(self, other):
        return (
            self.the_type == other.the_type and 
            self.x_absolute == other.x_absolute and 
            self.y_absolute == other.y_absolute and 
            self.last_seen == other.last_seen
        )
    
    def type_is_compatible(self, other):
        if self.the_type in TEAMMATE_TYPES:
            return (other.the_type not in OPPONENT_TYPES)
        elif self.the_type in OPPONENT_TYPES:
            return (other.the_type not in TEAMMATE_TYPES)
        else:
            return True
    
    def absolute_distance(self, other):
        dx = other.x_absolute - self.x_absolute
        dy = other.y_absolute - self.y_absolute
        return math.sqrt(dx**2 + dy**2)

def get_filtered_obstacles(obstacles, obstacle_age_threshold):
    # filter: only consider recent detections
    # but...
    # turns out the robot was instructed to send the absolute timestamp
    # of last detection of each obstacle, but no way to get the "obstacle age"
    # oh well, I'll do the best I can and exclude obstacles that are much older
    # than the most recent one.
    most_recent_obstacle_seen = -1
    for obs in obstacles:
        most_recent_obstacle_seen = max(most_recent_obstacle_seen, obs.last_seen)
    
    # actually apply recency filter, and another
    # filter: only consider robots (not goalposts)
    filtered_obstacles = []
    for obs in obstacles:
        obstacle_age_approx = most_recent_obstacle_seen - obs.last_seen
        if obstacle_age_approx < obstacle_age_threshold and obs.the_type in ROBOT_TYPES:
            filtered_obstacles.append(obs)
    
    return filtered_obstacles



def find_perfect_match_idx(obs, prev_obstacles, untracked_prev_idxs):
    for i in untracked_prev_idxs:
        if obs.is_perfect_match(prev_obstacles[i]):
            return i
    return None

class HungarianTracker:
    def __init__(self, first_obstacle_id):
        self.next_new_id = first_obstacle_id
        self.prev_obstacles = None
    
    def _get_new_id(self):
        i = self.next_new_id
        self.next_new_id += 1
        return i

    def _track_with_prev_obstacles(self, obstacles, prev_obstacles):
        # avoid side effect and preserve order
        untracked_idxs = list(range(len(obstacles)))
        untracked_prev_idxs = list(range(len(prev_obstacles)))
        tracked_obstacles = deepcopy(obstacles)

        # first step: perfectly matching records are automatically paired,
        # since it means they haven't been updated since last time
        for i in untracked_idxs:
            obs = obstacles[i]
            other_i = find_perfect_match_idx(obs, prev_obstacles, untracked_prev_idxs)
            if other_i is not None:
                tracked_obstacles[i].the_id = prev_obstacles[other_i].the_id
                untracked_idxs.remove(i)
                untracked_prev_idxs.remove(other_i)
            else:
                i += 1
        
        # next: actual, distance-based tracking

        # preliminary: cost calculation
        # gotta be VERY careful with the double set of indices,
        # so let's give a nonsensical name to the set furthest removed from the obstacles list
        # and stick to this convention like it's a type
        dm_rows = len(untracked_idxs)       # this is funya team
        dm_cols = len(untracked_prev_idxs)  # this is rinpa team
        distance_matrix = np.full((dm_rows, dm_cols), INVALID_WEIGHT)
        for funya in range(dm_rows):
            i = untracked_idxs[funya]
            for rinpa in range(dm_cols):
                j = untracked_prev_idxs[rinpa]
                # coherence constraint on obstacle type:
                # teammates cannot be matched to opponents and vice versa
                if obstacles[i].type_is_compatible(prev_obstacles[j]):
                    dist = obstacles[i].absolute_distance(prev_obstacles[j])
                    # distance constraint: obstacles further than some amount
                    # cannot be matched at all
                    if dist <= TRACKING_EXCLUSION_THRESHOLD:
                        distance_matrix[funya,rinpa] = dist
        # do the hungarian matching.
        # this version works for non-square matrices, but only of the rows<=cols kind
        # let's keep the branch as short as possible
        funyarinpa_matching_pairs = []  # name reminds the order to any readers who don't know what the hell is a funyarinpa
        if dm_rows <= dm_cols:
            # all good, we get: funya = assignments[rinpa], with -1 meaning no assignment
            assignments = hungarian(distance_matrix)
            for rinpa in range(len(assignments)):
                funya = assignments[rinpa]
                funyarinpa_matching_pairs.append((funya,rinpa))
        else:
            # gotta transpose the matrix, and thus we get: rinpa = assignments[funya], with -1 meaning no assignment
            assignments = hungarian(distance_matrix.transpose())
            for funya in range(len(assignments)):
                rinpa = assignments[funya]
                funyarinpa_matching_pairs.append((funya,rinpa))
        # convert funyarinpas to proper indices
        idx_matching_pairs = []
        for funya,rinpa in funyarinpa_matching_pairs:
            # these checks are all here rather than made while constructing funyarinpa_matching_pairs
            # in order to shorten the branching part
            # -1 means no assignment from hungarian
            if funya >= 0 and rinpa >= 0:
                # exclude invalid matches (hungarian still matches as much as it can,
                # even INVALID_WEIGHTs once it's exhausted all other options)
                if distance_matrix[funya,rinpa] < INVALID_WEIGHT:
                    i = untracked_idxs[funya]
                    other_i = untracked_prev_idxs[rinpa]
                    idx_matching_pairs.append((i, other_i))
        # now we're done with the double index set forever
        # assign ids to matched obstacles
        for i, other_i in idx_matching_pairs:
            tracked_obstacles[i].the_id = prev_obstacles[other_i].the_id
            untracked_idxs.remove(i)  # removal here is why I need this second loop
            untracked_prev_idxs.remove(other_i)  # same here

        # final: unmatched obstacles get assigned a new id
        for i in untracked_idxs.copy():
            tracked_obstacles[i].the_id = self._get_new_id()
            untracked_idxs.remove(i)

        assert len(untracked_idxs)==0

        return tracked_obstacles
    
    def track(self, obstacles):
        if self.prev_obstacles is None:
            tracked_obstacles = deepcopy(obstacles)
            for obs in tracked_obstacles:
                obs.the_id = self._get_new_id()
        else:
            tracked_obstacles = self._track_with_prev_obstacles(obstacles, self.prev_obstacles)
        
        # assert that we have assigned everything uniquely
        assert (
            all([obs.the_id!=UNASSIGNED_ID for obs in tracked_obstacles]) and
            all([
                tracked_obstacles[i].the_id != tracked_obstacles[j].the_id
                for i,j
                in np.ndindex(len(tracked_obstacles), len(tracked_obstacles))
                if j>i
            ])
        )
        
        self.prev_obstacles = tracked_obstacles
        return tracked_obstacles
