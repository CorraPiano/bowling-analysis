import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import medfilt



# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def clear_outside_frame_range(df, min_frame=50, max_frame=200):
    """
    Sets 'X' and 'Y' to NaN for frames outside the specified range.
    """
    df.loc[(df['Frame'] < min_frame) | (df['Frame'] > max_frame), ['X', 'Y']] = np.nan
    return df


def remove_outliers(df, threshold=0.1):
    """
    Removes outliers based on the distance between consecutive points.
    """
    df[['X', 'Y']] = df[['X', 'Y']].apply(pd.to_numeric, errors='coerce')
    
    distances = np.sqrt(df['X'].diff()**2 + df['Y'].diff()**2)
    outlier_threshold = distances.median() + threshold * distances.std()
    
    df_cleaned = df[distances <= outlier_threshold].copy().reset_index(drop=True)
    df_cleaned[['X', 'Y']] = df_cleaned[['X', 'Y']].round().astype(int)
    
    return df_cleaned


def apply_median_filter(df, kernel_size=5):
    """
    Applies a median filter to smooth the coordinates.
    """
    df['X'] = medfilt(df['X'], kernel_size=kernel_size)
    df['Y'] = medfilt(df['Y'], kernel_size=kernel_size)
    return df


def apply_savitzky_golay_filter(df, window_length=25, polyorder=1):
    """
    Applies a Savitzky-Golay filter to smooth the coordinates.
    """
    df['X'] = savgol_filter(df['X'], window_length=window_length, polyorder=polyorder).round().astype(int)
    df['Y'] = savgol_filter(df['Y'], window_length=window_length, polyorder=polyorder).round().astype(int)
    return df


def interpolate_missing_coordinates(df, start_frame=50, end_frame=200):
    """
    Interpolates missing coordinates within the given frame range.
    """
    all_frames = pd.DataFrame({'Frame': range(start_frame, end_frame + 1)})
    df_full = pd.merge(all_frames, df, on='Frame', how='left')
    
    df_full[['X', 'Y']] = df_full[['X', 'Y']].interpolate(method='linear').bfill().ffill()
    
    return apply_savitzky_golay_filter(df_full)


# ==============================================================================
#                            VIDEO PROCESSING FUNCTIONS
# ==============================================================================

def process_coordinates(input_csv: str, output_csv: str):
    """
    Processes the input CSV file by applying filtering, smoothing, and interpolation.
    """
    df = pd.read_csv(input_csv)
    df = clear_outside_frame_range(df)
    df = remove_outliers(df)
    df = apply_median_filter(df)
    df = interpolate_missing_coordinates(df)
    df.to_csv(output_csv, index=False)

    print(f"New circle positions saved to {output_csv}.")


# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    #PROJECT_ROOT = Path().resolve().parent.parent
    #INPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / "Circle_positions_2.0.csv")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "circle_positions" / "Circle_positions_2.0_clean.csv")
    
    process_coordinates(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
