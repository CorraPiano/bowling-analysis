import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def transform_coordinates(input_csv: str, output_csv: str, scale_x: float, scale_y: float):
    """
    Transforms the coordinates from the input CSV file and saves them to the output CSV file.
    """
    df = pd.read_csv(input_csv)
    transformed_points = []
    
    for _, row in df.iterrows():
        x_old, y_old = row['x'], row['y']
        x_new = int(x_old * scale_x)
        y_new = int(y_old * scale_y) + 65
        transformed_points.append([int(row['frame']), x_new, y_new])
    
    transformed_df = pd.DataFrame(transformed_points, columns=['frame', 'x', 'y'])
    transformed_df.to_csv(output_csv, index=False)

    print(f"Transformed coordinates have been saved to: {output_csv}")

def Savitzky_Golay_filter(df, window_length=60, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth the x and y coordinates in the DataFrame.
    """
    df = df.copy()
    df['x'] = savgol_filter(df['x'], window_length=window_length, polyorder=polyorder)
    df['y'] = savgol_filter(df['y'], window_length=window_length, polyorder=polyorder)
    df['x'] = df['x'].round().astype(int)
    df['y'] = df['y'].round().astype(int)

    return df


# ==============================================================================
#                           RECONSTRUCTION FUNCTIONS
# ==============================================================================

def process_reconstruction_deformed(input_csv: str, output_csv: str, template_path: str):
    """
    Process the reconstruction by transforming coordinates and applying smoothing.
    """
    df = pd.read_csv(input_csv)
    template = cv2.imread(template_path)
    h, w = template.shape[:2]

    old_width, old_height = 106, 1829
    new_width, new_height = w, h - 65

    scale_x = new_width / old_width
    scale_y = new_height / old_height

    transform_coordinates(input_csv, output_csv, scale_x, scale_y)

    df = pd.read_csv(output_csv)
    df_smooted = Savitzky_Golay_filter(df)
    df_smooted.to_csv(output_csv, index=False)