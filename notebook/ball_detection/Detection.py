import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import pandas as pd


# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def setup_output_video(cap: cv2.VideoCapture, output_video: str) -> cv2.VideoWriter:
    """
    Initializes and returns a VideoWriter object for saving processed frames.
    
    Parameters:
        cap (cv2.VideoCapture): VideoCapture object of the input video.
        output_video (str): Path to save the output video.
    
    Returns:
        cv2.VideoWriter: Configured video writer object.
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))


def initialize_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
    """
    Initializes and returns a Background Subtractor using MOG2 algorithm.
    
    Returns:
        cv2.BackgroundSubtractorMOG2: Configured background subtractor.
    """
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    bg_subtractor.setVarThresholdGen(25)
    return bg_subtractor


def warm_up_background_subtractor(cap: cv2.VideoCapture, bg_subtractor: cv2.BackgroundSubtractorMOG2, warmup_frames: int = 40):
    """
    Warms up the background subtractor by applying it to a few initial frames.
    
    Parameters:
        cap (cv2.VideoCapture): VideoCapture object of the input video.
        bg_subtractor (cv2.BackgroundSubtractorMOG2): Initialized background subtractor.
        warmup_frames (int): Number of frames to use for background model initialization.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(max(total_frames - warmup_frames, 0)):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame during warm-up.")
            break
        bg_subtractor.apply(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame


def detect_and_draw_circles(frame: np.ndarray, blurred: np.ndarray, total_frames: int, frame_count: int):
    """
    Detects circles using the Hough Transform and overlays them onto the frame.
    
    Parameters:
        frame (np.ndarray): Original frame.
        blurred (np.ndarray): Foreground mask after background subtraction.
    
    Returns:
        tuple: (Processed frame, detected circle coordinates or None)
    """
    min = int((total_frames - frame_count)*0.20 + 10) # TODO: change with more accuracy
    max = int((total_frames - frame_count)*0.20 + 40) # TODO: change with more accuracy

    # For recording_4:
    # min = int((total_frames - frame_count)*0.10 + 0)
    # max = int((total_frames - frame_count)*0.20 + 30)

    # For recording_2:
    # min = int((total_frames - frame_count)*0.20 + 10)
    # max = int((total_frames - frame_count)*0.20 + 40)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=min, maxRadius=max
    )
    
    output = frame.copy()
    circle_coords = None
    radius = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]  # Process only the first detected circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
        circle_coords = (x, y)
        radius = r

    return output, circle_coords, radius

def apply_absdiff(prev_frame: np.ndarray, frame: np.ndarray):
    """
    Applies absolute difference technique.

    Returns:
        bg_diff (np.ndarray): Processed frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    background = prev_gray.copy().astype("float")
    bg_diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))
    
    return bg_diff

def apply_morphological_operations(fg_mask: np.ndarray):
    """
    Applies morphological operations to clean up the foreground mask.
    
    Parameters:
        fg_mask (np.ndarray): Foreground mask after background subtraction.
    
    Returns:
        blurred (np.ndarray): Processed frame
    """
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    blurred = cv2.GaussianBlur(fg_mask, (9, 9), 2)

    return blurred

def apply_morphological_operations_lite(fg_mask: np.ndarray):
    """
    Applies morphological operations to clean up the foreground mask.
    
    Parameters:
        fg_mask (np.ndarray): Foreground mask after background subtraction.
    
    Returns:
        thresh (np.ndarray): Processed frame
    """
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    return thresh

def define_edges(blurred: np.ndarray, frame: np.ndarray):
    """
    Applies adaptive thresholding and edge detection.

    Returns:
        edges (np.ndarray): Processed frame
    """
    # Adaptive thresholding to highlight moving objects
    _, newThresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(newThresh, (15, 15), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask = newThresh)
    edges = cv2.Canny(masked_gray, 50, 150)

    return edges

def remove_background(filename, image):
    """
    Removes what is not the track in the image.

    Returns:
        result (np.ndarray): Processed frame
    """
    points = pd.read_csv(filename).values[:, :2].astype(np.int32)
    sorted_indices = np.argsort(points[:, 1])  # Sort by Y values
    points[sorted_indices[:2], 1] -= 100  # Move the two lowest points up by 25 pixels
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

# ==============================================================================
#                            VIDEO PROCESSING FUNCTIONS
# ==============================================================================

def process_video_with_background_subtractor(input_video: str, output_video: str, output_csv: str):
    """
    Processes an input video to detect and highlight moving circular objects, 
    and saves their center coordinates to a CSV file.
    
    Parameters:
        input_video (str): Path to the input video.
        output_video (str): Path to save the output video.
        output_csv (str): Path to save the circle coordinates.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    out = setup_output_video(cap, output_video)
    bg_subtractor = initialize_background_subtractor()
    warm_up_background_subtractor(cap, bg_subtractor)
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y"])  # Write CSV header
    
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video.")
                break
            
            fg_mask = bg_subtractor.apply(frame)
            blurred = apply_morphological_operations(fg_mask)

            processed_frame, circle_coords = detect_and_draw_circles(frame, blurred, total_frames, frame_count)
            out.write(processed_frame)
            
            x, y = circle_coords if circle_coords else (None, None)
            writer.writerow([frame_count, x, y])  # Save coordinates to CSV
            
            frame_count += 1
    
    print(f"Processed {frame_count} frames. \nCircle positions saved to {output_csv}.")
    cap.release()
    out.release()


def process_video_with_absdiff(input_video: str, input_points: str, output_video: str, output_csv: str):
    """
    Processes an input video to detect and highlight moving circular objects, 
    and saves their center coordinates to a CSV file.
    
    Parameters:
        input_video (str): Path to the input video.
        output_video (str): Path to save the output video.
        output_csv (str): Path to save the circle coordinates.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    out = setup_output_video(cap, output_video)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y", "Radius"])  # Write CSV header
    
        while frame_count < total_frames - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            prev_ret, prev_frame = cap.read()
            ret, frame = cap.read()
            if not prev_ret or not ret:
                print("Error: Could not read frame from video.")
                break

            bg_diff = apply_absdiff(prev_frame, frame)
            blurred = apply_morphological_operations_lite(bg_diff)
            # edged = define_edges(blurred, frame)
            clean_image = remove_background(input_points, blurred)

            processed_frame, circle_coords, radius = detect_and_draw_circles(frame, clean_image, total_frames, frame_count)
            out.write(processed_frame)
            
            x, y = circle_coords if circle_coords else (None, None)
            r = radius if radius else None
            writer.writerow([frame_count, x, y, r])  # Save coordinates to CSV
            
            frame_count += 1
    
    print(f"Processed {frame_count} frames. \nCircle positions saved to {output_csv}. \nVideo with ball detection saved to {output_video}")
    cap.release()
    out.release()

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":

    #PROJECT_ROOT = Path().resolve().parent.parent
    #INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_2" / "Recording_2.mp4")
    #INPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "lane_points" / "lane_points_2_frame_100.csv")
    #OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_2" / "Output_detected_test_2.mp4")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / "Circle_positions_2.csv")
    
    #process_video_with_background_subtractor(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_CSV_PATH)
    process_video_with_absdiff(INPUT_VIDEO_PATH, INPUT_CSV_PATH, OUTPUT_VIDEO_PATH, OUTPUT_CSV_PATH)
