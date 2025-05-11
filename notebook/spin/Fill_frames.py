import cv2
import pandas as pd

def fill_frames(video_path: str, csv_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Error opening video file")

    frame_count = 0
    while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1

    print(f"Total frames in video: {frame_count}")

    cap.release()

    df_coords = pd.read_csv(csv_path)

    # Initialize with NaNs for x, y, radius
    full_df = pd.DataFrame({
        'frame': list(range(frame_count)),
        'x': [None]*frame_count,
        'y': [None]*frame_count,
        'radius': [None]*frame_count,
        'x_axis': [None]*frame_count,
        'y_axis': [None]*frame_count,
        'z_axis': [None]*frame_count,
        'angle': [None]*frame_count,
    })

    # Set 'frame' as index for easier alignment
    df_coords.set_index('frame', inplace=True)
    full_df.set_index('frame', inplace=True)

    # Update the full_df with existing values from df_coords
    full_df.update(df_coords)

    # Reset index to save as CSV
    full_df.reset_index(inplace=True)

    # === Step 5: Save to new CSV ===
    full_df.to_csv(output_path, index=False)

    print(f"Saved updated CSV with all {frame_count} frames to {output_path}")