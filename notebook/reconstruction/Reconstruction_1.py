import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def compute_homography(input_csv_lane: str, width: int, height: int):
    """
    Computes the homography matrix using lane points from the input CSV file.
    """
    df = pd.read_csv(input_csv_lane)
    pts_src = df[['x', 'y']].values.astype(np.int32)

    pts_dst = np.array([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H

def apply_homography(input_csv_ball: str, H: np.ndarray, output_csv: str):
    """
    Applies homography transformation to ball positions from the input CSV file.
    """
    df = pd.read_csv(input_csv_ball)
    transformed_data = []

    for _, row in df.iterrows():
        frame_id, x, y, r = row['Frame'], row['X'], row['Y'], row['Radius']
        
        lower_intersection = np.array([[int(x), int(y + r)]], dtype=np.float32)
        
        transformed_point = cv2.perspectiveTransform(np.array([lower_intersection]), H)[0][0]
        
        transformed_data.append([frame_id, int(transformed_point[0]), int(transformed_point[1])])

    transformed_df = pd.DataFrame(transformed_data, columns=['Frame', 'Transformed_X', 'Transformed_Y'])
    transformed_df.to_csv(output_csv, index=False)
    print(f"Transformed data saved to {output_csv}.")

# ==============================================================================
#                           RECONSTRUCTION FUNCTIONS
# ==============================================================================


def process_reconstruction(input_csv_lane: str, input_csv_ball: str, output_csv: str):
    """
    Processes lane and ball data to reconstruct transformed positions.
    """
    width = 1066
    height = 18290

    H = compute_homography(input_csv_lane, width, height)
    apply_homography(input_csv_ball, H, output_csv)

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    #PROJECT_ROOT = Path().resolve().parent.parent
    #CSV_POSITIONS_FILE_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "lane_points" / "lane_points_2_frame_100.csv")
    #VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_2" / "Recording_2.mp4")
    #CSV_POINTS_POSITIONS_FILE_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / "Circle_positions_2.0_clean_radius.csv")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "reconstructed_positions" / "transformed_positions_2.csv")

    process_reconstruction(CSV_POSITIONS_FILE_PATH, CSV_POINTS_POSITIONS_FILE_PATH, OUTPUT_CSV_PATH)
