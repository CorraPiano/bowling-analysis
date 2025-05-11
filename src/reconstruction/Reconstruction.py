import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def compute_homographies_per_frame(input_csv_lane: str, width: int, height: int):
    """
    Computes a homography matrix for each frame using lane points from the input CSV file.
    Returns a dictionary: {frame_number: homography_matrix}
    """
    df = pd.read_csv(input_csv_lane)
    homographies = {}

    for _, row in df.iterrows():
        frame = int(row['Frame'])
        pts_src = np.array([
            [row['bottom_left_x'], row['bottom_left_y']],
            [row['bottom_right_x'], row['bottom_right_y']],
            [row['up_right_x'], row['up_right_y']],
            [row['up_left_x'], row['up_left_y']]
        ], dtype=np.float32)

        pts_dst = np.array([
            [0, height],
            [width, height],
            [width, 0],
            [0, 0]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(pts_src, pts_dst)
        if H is not None:
            homographies[frame] = H

    return homographies

def apply_homography_per_frame(input_csv_ball: str, homographies: dict, output_csv: str):
    """
    Applies per-frame homography transformations to ball positions.
    """
    df = pd.read_csv(input_csv_ball)
    transformed_data = []

    for _, row in df.iterrows():
        frame_id = int(row['frame'])
        x, y, r = row['x'], row['y'], row['radius']

        if pd.isna(x) or pd.isna(y) or pd.isna(r):
            transformed_data.append([frame_id, np.nan, np.nan])
            continue

        H = homographies.get(frame_id)
        if H is None:
            continue

        point = np.array([[[x, y + r]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, H)[0][0]
        # transformed_data.append([frame_id, int(transformed_point[0]), int(transformed_point[1])])
        transformed_data.append([frame_id, transformed_point[0], transformed_point[1]])

    transformed_df = pd.DataFrame(transformed_data, columns=['frame', 'x', 'y'])
    transformed_df.to_csv(output_csv, index=False)
    print(f"Transformed data saved to {output_csv}.")


# ==============================================================================
#                           RECONSTRUCTION FUNCTIONS
# ==============================================================================

def process_reconstruction(input_csv_lane: str, input_csv_ball: str, output_csv: str):
    """
    Processes lane and ball data to reconstruct transformed positions with per-frame homographies.
    """
    width = 106
    height = 1829

    homographies = compute_homographies_per_frame(input_csv_lane, width, height)
    apply_homography_per_frame(input_csv_ball, homographies, output_csv)
