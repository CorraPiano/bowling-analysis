import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
#                            CONFIGURATION PARAMETERS
# ==============================================================================

VIDEO_NUM = "7"
CONSISTENCY_THRESHOLD = 5
POSITION_TOLERANCE = 35
RADIUS_TOLERANCE = 10
ROI_MARGIN = 50
MAX_LOST_FRAMES = 2

# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def get_points_for_frame(df: pd.DataFrame, frame_number: int) -> np.ndarray:
    '''
    Retrieves lane points for a specific frame from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing lane points.
        frame_number (int): Frame number to retrieve points for.
    Returns:
        np.ndarray: Array of points for the specified frame.
    '''
    row = df[df['Frame'] == frame_number]
    if row.empty:
        raise ValueError(f"No points found for frame {frame_number}")
    return np.array([
        [row['bottom_left_x'].values[0], row['bottom_left_y'].values[0]],
        [row['bottom_right_x'].values[0], row['bottom_right_y'].values[0]],
        [row['up_right_x'].values[0], row['up_right_y'].values[0]],
        [row['up_left_x'].values[0], row['up_left_y'].values[0]]
    ], dtype=np.int32)

def compute_modified_polygon(points: np.ndarray) -> np.ndarray:
    '''
    Computes a modified polygon based on the top two points of the input points.
    Args:
        points (np.ndarray): Array of points representing the polygon.
    Returns:
        np.ndarray: Modified polygon points.
    '''
    top_indices = np.argsort(points[:, 1])[:2]
    top_points = points[top_indices]
    left_top, right_top = sorted(top_points, key=lambda pt: pt[0])

    left_top_mod = [left_top[0] - 50, left_top[1] - 70]
    right_top_mod = [right_top[0] + 50, right_top[1] - 70]

    return np.array([
        left_top_mod if np.array_equal(pt, left_top) else
        right_top_mod if np.array_equal(pt, right_top) else pt
        for pt in points
    ], dtype=np.int32)

def compute_upper_polygon(points: np.ndarray) -> np.ndarray:
    '''
    Computes a modified polygon based on the top two points of the input points.
    Args:
        points (np.ndarray): Array of points representing the polygon.
    Returns:
        np.ndarray: Modified polygon points.
    '''
    top_indices = np.argsort(points[:, 1])[:2]
    top_points = points[top_indices]
    left_top, right_top = sorted(top_points, key=lambda pt: pt[0])
    
    left_top_mod = [left_top[0] - 500, left_top[1] + 150]                                   # TODO: calculate it in another way (proportions between the lines?)
    right_top_mod = [right_top[0] + 150, right_top[1] + 150]                                # TODO: calculate it in another way (proportions between the lines?)

    return np.array([
        left_top_mod if np.array_equal(pt, left_top) else
        right_top_mod if np.array_equal(pt, right_top) else pt
        for pt in points
    ], dtype=np.int32)

def compute_approx_radius(df: pd.DataFrame) -> int:
    '''
    Computes the approximate radius of the ball based on the distance between two points.
    Args:
        df (pd.DataFrame): DataFrame containing lane points.
    Returns:
        int: Approximate radius of the ball.
    '''
    points = get_points_for_frame(df, 0)
    radius = int(np.linalg.norm(points[0] - points[1]) / 10)
    return radius

def remove_background(frame_number: int, df: pd.DataFrame, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Removes the background from the image using a polygon mask.
    Args:
        frame_number (int): Frame number to retrieve points for.
        df (pd.DataFrame): DataFrame containing lane points.
        image (np.ndarray): Input image.
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the masked image and the polygon points.
    '''
    points = get_points_for_frame(df, frame_number)
    polygon = compute_modified_polygon(points)
    mask = cv2.fillPoly(np.zeros(image.shape[:2], dtype=np.uint8), [polygon], 255)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, polygon

def preprocess_roi(image: np.ndarray) -> np.ndarray:
    '''
    Preprocesses the image for circle detection.
    Args:
        image (np.ndarray): Input image.
    Returns:
        np.ndarray: Preprocessed image.
    '''
    return cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)

def detect_circle(preprocessed_img: np.ndarray, r_approx: int) -> np.ndarray | None:
    '''
    Detects circles in the preprocessed image using Hough Circle Transform.
    Args:
        preprocessed_img (np.ndarray): Preprocessed image.
    Returns:
        np.ndarray | None: Detected circles or None if no circles are found.
    '''
    min_radius = int(r_approx * 0.65)
    max_radius = int(r_approx)
    return cv2.HoughCircles(
        preprocessed_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius
    )

def detect_circle_after_roi(preprocessed_img: np.ndarray, r_approx: int) -> np.ndarray | None:
    '''
    Detects circles in the preprocessed image using Hough Circle Transform.
    Args:
        preprocessed_img (np.ndarray): Preprocessed image.
    Returns:
        np.ndarray | None: Detected circles or None if no circles are found.
    '''
    min_radius = max(0, int(r_approx*0.7))
    max_radius = int(r_approx*1.1)
    return cv2.HoughCircles(
        preprocessed_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius
    )

def detect_circle_roi(preprocessed_img: np.ndarray, r_approx: int) -> np.ndarray | None:
    '''
    Detects circles in the preprocessed image using Hough Circle Transform with a region of interest.
    Args:
        preprocessed_img (np.ndarray): Preprocessed image.
        frame_number (int): Current frame number.
        r_approx (int): Approximate radius of the circle.
    Returns:
        np.ndarray | None: Detected circles or None if no circles are found.
    '''
    min_radius = max(0, int(r_approx*0.89))
    max_radius = int(r_approx*1.08)
    return cv2.HoughCircles(
        preprocessed_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=500,
        param1=200,
        param2=5,
        minRadius=min_radius,
        maxRadius=max_radius
    )

def update_roi(x: int, y: int, r: int, frame_shape, base_margin: int, scale_factor: float) -> tuple[int, int, int, int]:
    '''
    Updates the region of interest (ROI) based on the detected circle.
    Args:
        x (int): X-coordinate of the circle center.
        y (int): Y-coordinate of the circle center.
        r (int): Radius of the circle.
        frame_shape (tuple): Shape of the video frame.
        base_margin (int): Base margin around the circle.
        scale_factor (float): Scale factor for dynamic margin.
    Returns:
        tuple[int, int, int, int]: Updated ROI coordinates (x_min, y_min, x_max, y_max).
    '''
    height, width = frame_shape[:2]
    dynamic_margin = max(int(r * scale_factor), 10)
    x_min = max(0, x - r - dynamic_margin)
    y_min = max(0, y - r - dynamic_margin)
    x_max = min(width, x + r + dynamic_margin)
    y_max = min(height, y + r + 3)
    return (x_min, y_min, x_max, y_max)

# ==============================================================================
#                            VIDEO PROCESSING FUNCTIONS
# ==============================================================================

def process_video_with_roi(input_video: str, input_points: str, output_video: str, output_csv: str):
    '''
    Processes the video with a region of interest (ROI) for circle detection.
    Args:
        input_video (str): Path to the input video.
        input_points (str): Path to the CSV file with lane points.
        output_video (str): Path to save the output video.
        output_csv (str): Path to save the circle positions CSV.
    '''
    # --- Load Data ---
    df = pd.read_csv(input_points)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise IOError("Error opening video file!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # --- State Variables ---
    last_circle = None
    consistency_counter = 0
    lost_counter = 0
    use_roi = False
    search_roi = None
    frame_idx = 0
    first_time = True
    scale_factor = 0.6
    r_approx = None

    # --- CSV Results Storage ---
    circle_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        try:
            masked_frame, _ = remove_background(frame_idx, df, frame)

            if consistency_counter < CONSISTENCY_THRESHOLD and first_time:
                upper_polygon = compute_upper_polygon(get_points_for_frame(df, frame_idx))
                upper_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(upper_mask, [upper_polygon], 255)
                search_region = cv2.bitwise_and(masked_frame, masked_frame, mask=upper_mask)
                roi_offset = (0, 0)
                preprocessed = preprocess_roi(search_region)
                r_approx = compute_approx_radius(df)
                circles = detect_circle(preprocessed, r_approx)
            elif use_roi and search_roi:
                x_min, y_min, x_max, y_max = search_roi
                search_region = masked_frame[y_min:y_max, x_min:x_max]
                roi_offset = (x_min, y_min)
                preprocessed = preprocess_roi(search_region)
                circles = detect_circle_roi(preprocessed, r_approx)
            else:
                search_region = masked_frame
                roi_offset = (0, 0)
                preprocessed = preprocess_roi(search_region)
                circles = detect_circle_after_roi(preprocessed, r_approx)

            valid_circle = None

            if circles is not None:
                x, y, r = np.round(circles[0, 0]).astype("int")
                x_global, y_global, r_global = x + roi_offset[0], y + roi_offset[1], r
                valid_circle = (x_global, y_global, r_global)

            if valid_circle:
                dx, dy, dr = np.abs(np.subtract(valid_circle, last_circle)) if last_circle else (0, 0, 0)
                if last_circle and dx < POSITION_TOLERANCE and dy < POSITION_TOLERANCE and dr < RADIUS_TOLERANCE:
                    consistency_counter += 1
                else:
                    consistency_counter = 1

                last_circle = valid_circle
                lost_counter = 0

                cv2.circle(frame, (valid_circle[0], valid_circle[1]), valid_circle[2], (0, 255, 0), 2)
                cv2.circle(frame, (valid_circle[0], valid_circle[1]), 2, (0, 0, 255), 3)

                if consistency_counter >= CONSISTENCY_THRESHOLD:
                    search_roi = update_roi(x_global, y_global, r_global, frame.shape, ROI_MARGIN, scale_factor)
                    r_approx = r_global
                    first_time = False
                    use_roi = True

                # Save valid circle to results
                circle_data.append({
                    'frame': frame_idx,
                    'x': valid_circle[0],
                    'y': valid_circle[1],
                    'radius': valid_circle[2]
                })
            else:
                lost_counter += 1
                if lost_counter >= MAX_LOST_FRAMES:
                    use_roi = False
                    consistency_counter = 0
                    last_circle = None
                    search_roi = None
                    #r_approx = None

                # Save empty row if no circle detected
                circle_data.append({
                    'frame': frame_idx,
                    'x': None,
                    'y': None,
                    'radius': None
                })

        except ValueError as e:
            print(f"Skipping frame {frame_idx}: {e}")
            circle_data.append({
                'frame': frame_idx,
                'x': None,
                'y': None,
                'radius': None
            })

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Save circle data to CSV
    pd.DataFrame(circle_data).to_csv(output_csv, index=False)
    print(f"Circle data saved to {output_csv} \nVideo saved to {output_video}")

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":

    #PROJECT_ROOT = Path().resolve().parent.parent
    #INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_2" / "Recording_2.mp4")
    #INPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "lane_points" / "lane_points_processed_2.csv")
    #OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_2" / "Ball_detected_raw_2.mp4")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / "Circle_positions_raw_2.csv")

    process_video_with_roi(INPUT_VIDEO_PATH, INPUT_CSV_PATH, OUTPUT_VIDEO_PATH, OUTPUT_CSV_PATH)
