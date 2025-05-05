import cv2
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
#                              CONSTANTS
# ==============================================================================

WIDTH = 106       # Approximate pixels for 1.0668m
HEIGHT = 1829     # Approximate pixels for 18.29m
BROWN_COLOR = (135, 184, 222)  # RGB for burly wood
BALL_COLOR = (0, 0, 255)
BALL_RADIUS = 10
LINE_THICKNESS = 2


# ==============================================================================
#                          AUXILIARY FUNCTIONS
# ==============================================================================

def load_positions_from_csv(csv_file_path):
    """
    Load ball positions from a CSV file.
    Returns:
        A dictionary with frame numbers as keys and (x, y) tuples as values.
    """
    positions = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                frame_str, x_str, y_str = row[:3]
                if frame_str.strip() and x_str.strip() and y_str.strip():
                    try:
                        frame_num = int(frame_str)
                        x = int(float(x_str))
                        y = int(float(y_str))
                        positions[frame_num] = (x, y)
                    except ValueError:
                        continue
    return positions


# ==============================================================================
#                           PRINCIPAL FUNCTION
# ==============================================================================

def main(input_video_path, csv_file_path, output_video_path):
    """
    Main function to read a video, overlay a trajectory from a CSV file, and save the output.
    """
    alley = np.full((HEIGHT, WIDTH, 3), BROWN_COLOR, dtype=np.uint8)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))

    positions = load_positions_from_csv(csv_file_path)
    trajectory = []
    frame_count = 0

    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break

        frame = alley.copy()

        if frame_count in positions:
            x, y = positions[frame_count]
            trajectory.append((x, y))

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], BALL_COLOR, LINE_THICKNESS)

        if frame_count in positions:
            cv2.circle(frame, (x, y), BALL_RADIUS, BALL_COLOR, -1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Tracking video saved to {output_video_path}")


# ==============================================================================
#                                ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    #INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUM}" / f"Recording_{VIDEO_NUM}.mp4")
    #CSV_FILE_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "reconstructed_positions" / f"Transformed_positions_processed_{VIDEO_NUM}.csv")
    #OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUM}" / f"Reconstructed_trajectory_processed_{VIDEO_NUM}.mp4")

    main(INPUT_VIDEO_PATH, CSV_FILE_PATH, OUTPUT_VIDEO_PATH)
