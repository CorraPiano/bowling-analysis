import cv2
import csv

# ==============================================================================
#                              CONSTANTS
# ==============================================================================

BALL_COLOR = (0, 0, 255)
BALL_RADIUS = 10
LINE_THICKNESS = 5

# ==============================================================================
#                          AUXILIARY FUNCTIONS
# ==============================================================================

def load_positions_from_csv(csv_file_path):
    """
    Load (frame_num, x, y) positions from a CSV file.
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

def trajectory_on_reconstruction_deformed(input_video_path, csv_file_path, csv_template_path, output_video_path):
    """
    Main function to read a video, overlay a trajectory from a CSV file, and save the output.
    """
    template = cv2.imread(csv_template_path)
    height, width = template.shape[:2]

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    positions = load_positions_from_csv(csv_file_path)

    trajectory = []
    frame_count = 0

    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break

        frame = template.copy()

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
