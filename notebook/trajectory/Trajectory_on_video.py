import cv2
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def load_video(path: str):
    """
    Load a video file using OpenCV.
    Returns:
        cap: OpenCV VideoCapture object.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Error: Could not open video.")
    return cap


def get_video_properties(cap):
    """
    Get video properties such as width, height, and frames per second (fps).
    Returns:
        width: Width of the video frames.
        height: Height of the video frames.
        fps: Frames per second of the video.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def create_video_writer(path: str, width: int, height: int, fps: int):
    """
    Create a VideoWriter object to save the output video.
    Returns:
        out: OpenCV VideoWriter object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def load_lane_homographies(csv_path: str, width: int = 106, height: int = 1829):
    """
    Load lane points from a CSV file and compute homographies for each frame.
    Returns:
        homographies: A dictionary with frame numbers as keys and homography matrices as values.
    """
    lane_points = pd.read_csv(csv_path)
    homographies = {}

    for _, row in lane_points.iterrows():
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

        H, _ = cv2.findHomography(pts_dst, pts_src)
        if H is not None:
            homographies[frame] = H

    return homographies


def load_positions(csv_path: str):
    """
    Load (frame_num, x, y) positions from a CSV file.
    Returns:
        A dictionary with frame numbers as keys and (x, y) tuples as values.
    """
    positions = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for coord in reader:
            frame_num, x, y = coord
            frame_num = int(frame_num)
            if x and y:
                positions[frame_num] = (float(x), float(y))
            else:
                positions[frame_num] = None
    return positions


def draw_trajectory(cap, out, positions, homographies):
    """
    Draw the trajectory of the ball on the video frames.
    Returns:
        transformed_points: A list of transformed points (frame, x, y).
    """
    frame_count = 0
    trajectory = []
    transformed_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        if frame_count in positions and positions[frame_count] is not None:
            trajectory.append(positions[frame_count])

        refreshed_trajectory = []
        if frame_count in homographies:
            H = homographies[frame_count]
            for x, y in trajectory:
                transformed = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)[0][0]
                refreshed_trajectory.append((int(transformed[0]), int(transformed[1])))

        for i in range(1, len(refreshed_trajectory)):
            cv2.line(frame, refreshed_trajectory[i - 1], refreshed_trajectory[i], (0, 0, 255), 2)

        if refreshed_trajectory:
            x, y = refreshed_trajectory[-1]
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            if frame_count in positions and positions[frame_count] is not None:
                transformed_points.append([frame_count, x, y])

        out.write(frame)
        frame_count += 1

    return transformed_points


def save_transformed_points(points, output_csv_path):
    """
    Save the transformed points to a CSV file.
    """
    df = pd.DataFrame(points, columns=['frame', 'x', 'y'])
    df.to_csv(output_csv_path, index=False)

    print(f"Transformed coordinates have been saved to: {output_csv_path}")

# ==============================================================================
#                           PRINCIPAL FUNCTION
# ==============================================================================


def main(input_video: str, transformed_csv: str, lane_csv: str, output_video: str, output_csv: str):
    """
    Main function to read a video, overlay a trajectory from a CSV file, and save the output.
    """
    cap = load_video(input_video)
    frame_width, frame_height, fps = get_video_properties(cap)
    out = create_video_writer(output_video, frame_width, frame_height, fps)

    homographies = load_lane_homographies(lane_csv)
    positions = load_positions(transformed_csv)

    transformed_points = draw_trajectory(cap, out, positions, homographies)

    cap.release()
    out.release()

    print(f"Tracking video saved to {output_video}")
    save_transformed_points(transformed_points, output_csv)

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================


if __name__ == "__main__":

    #INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUM}" / f"Recording_{VIDEO_NUM}.mp4")
    #TRASFORMED_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "reconstructed_positions" / f"Transformed_positions_processed_{VIDEO_NUM}.csv")
    #LANE_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "lane_points" / f"Lane_points_{VIDEO_NUM}.csv")
    #OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUM}" / f"Tracked_output_{VIDEO_NUM}.mp4")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / f"Ball_lower_point_raw_{VIDEO_NUM}.csv")

    main(INPUT_VIDEO_PATH, TRASFORMED_CSV_PATH, LANE_CSV_PATH, OUTPUT_VIDEO_PATH, OUTPUT_CSV_PATH)
