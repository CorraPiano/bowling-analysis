from collections import deque
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import medfilt
from scipy.signal import savgol_filter

# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def remove_low_y_coordinates(df):
    """
    Remove low y-coordinates from the DataFrame.
    Returns:
    pd.DataFrame
        The DataFrame with low y-coordinates removed.
    """
    df_cleaned = df.copy()
    df_cleaned['y'] = pd.to_numeric(df_cleaned['y'], errors='coerce')
    mask = (df_cleaned['y'] > 1750) | (df_cleaned['y'] < 30)
    df_cleaned.loc[mask, ['x', 'y']] = np.nan

    return df_cleaned

def remove_low_y_coordinates_v2(df):
    """
    Remove low y-coordinates from the DataFrame.
    Returns:
    pd.DataFrame
        The DataFrame with low y-coordinates removed.
    """
    df_cleaned = df.copy()
    df_cleaned['y'] = pd.to_numeric(df_cleaned['y'], errors='coerce')
    low_y_count = 0
    last_five_y = deque(maxlen=4)

    for index, row in df.iterrows():
        last_five_y.append(row['y'])

        if len(last_five_y) == 4 and all(y < 110 for y in last_five_y):
            low_y_count += 1

        if low_y_count >= 4:
            df_cleaned.loc[index, ['x', 'y']] = np.nan

    return df_cleaned

def rolling_median_mad(values, window_size):
    """
    Calculate the rolling median and median absolute deviation (MAD) for a given window size.
    Returns:
    tuple
        Two numpy arrays containing the rolling median and MAD values.
    """
    median_values = []
    mad_values = []

    window = deque(maxlen=window_size)

    for value in values:
        window.append(value)
        if len(window) == window_size:
            median = np.median(window)
            mad = np.median(np.abs(np.array(window) - median))
            median_values.append(median)
            mad_values.append(mad)
        else:
            median_values.append(np.nan)
            mad_values.append(np.nan)

    return np.array(median_values), np.array(mad_values)


def remove_outliers_with_rolling(df: pd.DataFrame, threshold: float = 2.5, window_size: int = 2) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame using a rolling median and MAD method.
    Returns:
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    df_clean = df.copy()
    initial_nan_mask = df_clean[['x', 'y']].isna().any(axis=1)

    x_median, x_mad = rolling_median_mad(df_clean['x'].values, window_size)
    y_median, y_mad = rolling_median_mad(df_clean['y'].values, window_size)
    
    distances = np.sqrt((df_clean['x'].values - x_median) ** 2 + (df_clean['y'].values - y_median) ** 2)

    if np.nanmedian(np.abs(distances - np.nanmedian(distances))) == 0:
        return df_clean

    modified_z = 0.6745 * (distances - np.nanmedian(distances)) / np.nanmedian(np.abs(distances - np.nanmedian(distances)))
    mask_outliers = np.abs(modified_z) < threshold
    new_outlier_mask = ~mask_outliers & ~initial_nan_mask
    df_clean.loc[new_outlier_mask, ['x', 'y']] = np.nan

    return df_clean

def median_filter(df, kernel_size=3):
    """
    Apply a median filter to the x and y coordinates in the DataFrame.
    Returns:
    pd.DataFrame
        The DataFrame with median filtering applied.
    """
    df = df.copy()
    df['x'] = medfilt(df['x'], kernel_size=kernel_size)
    df['y'] = medfilt(df['y'], kernel_size=kernel_size)

    df = df[df['x'] > 0]
    df = df[df['y'] > 0] 
    return df

def Savitzky_Golay_filter(df, window_length=50, polyorder=3):
    """
    Apply a Savitzky-Golay filter to the x and y coordinates in the DataFrame.
    Returns:
    pd.DataFrame
        The DataFrame with Savitzky-Golay filtering applied.
    """
    df = df.copy()
    df['x'] = savgol_filter(df['x'], window_length=window_length, polyorder=polyorder)
    df['y'] = savgol_filter(df['y'], window_length=window_length, polyorder=polyorder)
    #df['x'] = df['x'].round().astype(int)
    #df['y'] = df['y'].round().astype(int)

    return df

def interpolate_missing_coordinates(df):
    """
    Interpolate missing coordinates in the DataFrame.
    Returns:
    pd.DataFrame
        The DataFrame with missing coordinates interpolated.
    """
    df = df.copy().set_index('frame')
    
    full_index = range(df.index.min(), df.index.max() + 1)
    df_full = df.reindex(full_index)
    
    df_full['x'] = df_full['x'].interpolate(method='linear')
    df_full['y'] = df_full['y'].interpolate(method='linear')
    
    df_full['x'] = df_full['x'].bfill().ffill()
    df_full['y'] = df_full['y'].bfill().ffill()
    
    df_full = df_full.reset_index().rename(columns={'index': 'frame'})
    df_full = Savitzky_Golay_filter(df_full)
    
    return df_full

# ==============================================================================
#                             PROCESSING FUNCTIONS
# ==============================================================================


def process_data(input_csv: str, output_csv: str):
    """
    Process the input CSV file and save the cleaned data to the output CSV file.
    """
    df_coords = pd.read_csv(input_csv)
    df_final = remove_low_y_coordinates_v2(df_coords)
    df_final = remove_low_y_coordinates(df_final)
    df_filtered = remove_outliers_with_rolling(df_final)
    df_smoothed = median_filter(df_filtered)
    df_interpolated = interpolate_missing_coordinates(df_smoothed)
    df_interpolated.to_csv(output_csv, index=False)

    print("Cleaned data saved to: ", output_csv)


# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    #PROJECT_ROOT = Path().resolve().parent.parent
    #INPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "ball_detection" / "intermediate_data" / f"Circle_positions_raw_{VIDEO_NUM}.csv")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "ball_detection" / "intermediate_data" / f"Circle_positions_cleaned_{VIDEO_NUM}.csv")

    process_data(CSV_POSITIONS_FILE_PATH, OUTPUT_CSV_PATH)