import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def visualize(fname):
    data = pd.read_csv(fname)
    frames = data.frame.unique()

    for frame in frames:
        frame_data = data.loc[data.frame == frame]

        ego_color = "red"
        ball_color = "grey"
        other_colors = ["green", "blue", "fuchsia", "cyan", "lime", "darkorange", "rebeccapurple"]
        for _, row in frame_data.iterrows():
            if row.id == row.ego_id:
                color = ego_color
            elif row.klasse == "ball":
                color = ball_color
            else:
                color = other_colors[row.id % len(other_colors)]
            plt.scatter(row.field_pos_x, row.field_pos_y, color=color)
        plt.xlim(-5000, 5000)
        plt.ylim(-3500, 3500)
        plt.show()

# i.e. a directory called "data" in the parent folder to this repo's directory
# note to self: parents list is ordered from direct father to root, so no need for negative indices
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

visualize(DATA_DIR / "processed_real_test_set" / "6.csv")
# visualize(DATA_DIR / "processed_by_gameegoid" / "10034-1305.csv")
