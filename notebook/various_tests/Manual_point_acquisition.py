import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# ==============================================================================
#                          POINT ACQUISITION FUNCTION
# ==============================================================================

# Global variable to store selected points
points = []

def on_mouse_click(event, x, y, flags, param):
    """
    Mouse callback function to capture the points when clicked.
    Draws a line and circles between two points when both are selected.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        if len(points) == 2:
            # Draw a line between the points
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            # Draw circles on the points
            cv2.circle(frame, points[0], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[1], 5, (0, 0, 255), -1)
            # Display the updated frame
            cv2.imshow("Frame with Points", frame)


def process_frame_for_points(frame):
    """
    Main function to process the frame, capture two points, and return them.
    """
    global points
    points = []  # Reset points for each new frame

    # Set up the OpenCV window and mouse callback
    cv2.imshow("Frame with Points", frame)
    cv2.setMouseCallback("Frame with Points", on_mouse_click)

    # Wait until two points are selected
    while len(points) < 2:
        cv2.waitKey(1)  # Wait for a key press and capture mouse events

    print(f"Selected Points: {points}")
    return points

# ==============================================================================
#                                  MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":

    '''
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_2" / "Recording_2.mp4")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    cap.release()
    '''
    process_frame_for_points(frame)