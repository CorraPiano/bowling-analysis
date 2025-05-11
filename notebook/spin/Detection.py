from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import math

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================

def load_video_and_data(video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Could not open video.")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Error: The CSV file is empty.")
    
    return cap, df

def roi_bounds(center, radius, frame_shape, offset=2):
    x_min = max(center[0] - radius - offset, 0)
    x_max = min(center[0] + radius + offset, frame_shape[1])
    y_min = max(center[1] - radius - offset, 0)
    y_max = min(center[1] + radius + offset, frame_shape[0])
    return x_min, x_max, y_min, y_max

def compute_optical_flow(gray1, gray2, mask, ball_radius):
    feature_params = dict(maxCorners=100, qualityLevel=0.001, minDistance=0, blockSize=3)
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))

    p0 = cv2.goodFeaturesToTrack(gray1, mask=mask, **feature_params)
    if p0 is None:
        raise ValueError("No features detected. Try relaxing parameters.")
    
    p1, status_forward, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    p0r, status_backward, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, p1, None, **lk_params)

    fb_error = np.linalg.norm(p0 - p0r, axis=2)
    return p0, p1, p0r, status_forward, status_backward, fb_error

def filter_3d_points(p0, p1, p0r, status_forward, status_backward, fb_error,
                     center_roi1, center_roi2, ball_radius, fb_threshold=10.0, low_threshold_factor=5):
    movement_threshold = ball_radius
    low_movement_threshold = ball_radius / low_threshold_factor

    old3d, new3d, good_pts = [], [], []
    for old_pt, new_pt, s1, s2, err in zip(p0.reshape(-1, 2), p1.reshape(-1, 2),
                                           status_forward.ravel(), status_backward.ravel(), fb_error.ravel()):
        if s1 and s2 and err < fb_threshold:
            displacement = np.linalg.norm(new_pt - old_pt)
            if low_movement_threshold < displacement < movement_threshold:
                ox, oy = old_pt - center_roi1
                nx, ny = new_pt - center_roi2
                oz = math.sqrt(max(ball_radius**2 - ox**2 - oy**2, 0))
                nz = math.sqrt(max(ball_radius**2 - nx**2 - ny**2, 0))
                old3d.append([ox, oy, oz])
                new3d.append([nx, ny, nz])
                good_pts.append((old_pt, new_pt))
    
    return np.array(old3d), np.array(new3d), good_pts

def compute_rotation(old3d, new3d):
    P, Q = old3d, new3d
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0):
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        axis /= np.linalg.norm(axis)

    return axis, theta

# ==============================================================================
#                               MAIN PROCESSING FUNCTION
# ==============================================================================

def process_spin(input_video_path, input_csv_path, output_csv_path):
    cap, df = load_video_and_data(input_video_path, input_csv_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offset = 2
    output_data = []

    for frame_number in range(frame_count - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        if not (ret1 and ret2):
            continue

        try:
            df_row1, df_row2 = df.iloc[frame_number], df.iloc[frame_number + 1]
            if df_row1[['x', 'y', 'radius']].isnull().any() or df_row2[['x', 'y', 'radius']].isnull().any():
                continue
        except IndexError:
            continue

        ball_center1 = np.array([int(df_row1['x']), int(df_row1['y'])])
        ball_center2 = np.array([int(df_row2['x']), int(df_row2['y'])])
        ball_radius = int(df_row1['radius'])

        x_min1, x_max1, y_min1, y_max1 = roi_bounds(ball_center1, ball_radius, frame1.shape, offset)
        x_min2, x_max2, y_min2, y_max2 = roi_bounds(ball_center2, ball_radius, frame2.shape, offset)

        roi1, roi2 = frame1[y_min1:y_max1, x_min1:x_max1], frame2[y_min2:y_max2, x_min2:x_max2]
        gray1, gray2 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

        center_roi1 = ball_center1 - [x_min1, y_min1]
        center_roi2 = ball_center2 - [x_min2, y_min2]

        mask = np.zeros_like(gray1)
        cv2.circle(mask, tuple(center_roi1.astype(int)), int(ball_radius * 0.8), 255, -1)

        try:
            p0, p1, p0r, s_f, s_b, fb_error = compute_optical_flow(gray1, gray2, mask, ball_radius)
        except ValueError:
            continue

        old3d, new3d, good_pts = filter_3d_points(p0, p1, p0r, s_f, s_b, fb_error,
                                                  center_roi1, center_roi2, ball_radius)

        if old3d.shape[0] < 3:
            # Retry with more lenient filtering
            old3d, new3d, good_pts = filter_3d_points(p0, p1, p0r, s_f, s_b, fb_error,
                                                      center_roi1, center_roi2, ball_radius,
                                                      low_threshold_factor=10)

        if old3d.shape[0] < 3:
            continue

        axis, theta = compute_rotation(old3d, new3d)

        output_data.append({
            "frame": int(df_row1['frame']),
            "x": df_row1['x'],
            "y": df_row1['y'],
            "radius": df_row1['radius'],
            "x_axis": axis[0],
            "y_axis": axis[1],
            "z_axis": axis[2],
            "angle": theta
        })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved rotation data to {output_csv_path}")

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    #video_number = "7"
    #PROJECT_ROOT = Path().resolve()
    #INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUMBER}" / f"Recording_{VIDEO_NUMBER}.mp4")
    #INPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / f"Adjusted_positions_new_{VIDEO_NUMBER}.csv")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "spin" / "intermediate_data" / f"Rotation_data_TEST_{VIDEO_NUMBER}.csv")
    
    process_spin(INPUT_VIDEO_PATH, INPUT_CSV_PATH, OUTPUT_CSV_PATH)
