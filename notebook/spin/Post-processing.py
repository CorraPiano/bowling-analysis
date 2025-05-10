import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d


# ==============================================================================
#                             OUTLIER REMOVAL FUNCTIONS
# ==============================================================================

def remove_outliers(df, threshold=0.50):
    df = df.copy()
    valid_mask = df['x_axis'].notna() & df['y_axis'].notna()
    valid_df = df[valid_mask]

    x_model = LinearRegression().fit(valid_df[['frame']], valid_df['x_axis'])
    y_model = LinearRegression().fit(valid_df[['frame']], valid_df['y_axis'])

    x_pred = x_model.predict(valid_df[['frame']])
    y_pred = y_model.predict(valid_df[['frame']])

    x_error = np.abs(valid_df['x_axis'] - x_pred)
    y_error = np.abs(valid_df['y_axis'] - y_pred)

    within_threshold = (x_error <= threshold) & (y_error <= threshold)
    outlier_indices = valid_df.index[~within_threshold]

    df.loc[outlier_indices, ['x_axis', 'y_axis', 'z_axis', 'angle']] = np.nan

    return df


def remove_angle_outliers(series, threshold=1):
    z_scores = (series - series.mean()) / series.std()
    series = series.copy()

    series.loc[abs(z_scores) > threshold] = np.nan
    return series


# ==============================================================================
#                             SCALING AND SMOOTHING
# ==============================================================================

def scale_x_axis(df):
    df = df.copy()
    scale_factors = np.linspace(df['x_axis'].iloc[0], 1 / df['x_axis'].iloc[-1], len(df))
    df['x_axis'] *= scale_factors
    return df


def scale_y_axis(df):
    df = df.copy()
    scale_factors = np.linspace(1, 0, len(df))
    df['y_axis'] *= scale_factors
    return df


def smooth_series(series, first_idx, last_idx, window=5):
    smoothed = series.copy()
    segment = series[first_idx:last_idx + 1].rolling(window=window, center=True).mean()
    segment = segment.interpolate(method='cubic', limit_direction='both').bfill().ffill()
    smoothed = series.copy()
    smoothed.loc[first_idx:last_idx + 1] = segment
    series.loc[first_idx:last_idx] = smoothed
    return series


# ==============================================================================
#                             TRANSFORMATION AND INTERPOLATION
# ==============================================================================

def interpolate_axes_from_existing(new_df, old_df):
    interpolated_df = new_df.copy()
    valid_frames = old_df.dropna(subset=['x_axis', 'y_axis', 'z_axis'])['frame'].unique()

    for axis in ['x_axis', 'y_axis', 'z_axis']:
        interpolated_series = new_df[axis].interpolate(method='linear', limit_direction='both')
        interpolated_df.loc[interpolated_df['frame'].isin(valid_frames), axis] = \
            interpolated_series[interpolated_df['frame'].isin(valid_frames)]

    return interpolated_df


def compute_z_axis_from_xy(df, y_axis_avg):
    df = df.copy()
    z_values = 1 - df['x_axis'] ** 2 - df['y_axis'] ** 2
    z_values[z_values < 0] = np.nan
    df['z_axis'] = np.sqrt(z_values) if y_axis_avg < 0 else -np.sqrt(z_values)
    return df


# ==============================================================================
#                             MAIN VIDEO PROCESSING
# ==============================================================================

def spin_detection(input_video_path, output_csv_path):
    df = pd.read_csv(input_video_path)
    z_axis_avg = df['z_axis'].mean()

    # Flip sign of axes based on z_axis
    flip_condition = df['z_axis'] > 0.25 if z_axis_avg < 0 else df['z_axis'] < 0.25
    df.loc[flip_condition, ['x_axis', 'y_axis', 'z_axis']] *= -1

    y_axis_avg = df['y_axis'].mean()

    df_y = df.copy()
    y_condition = df_y['y_axis'] > -y_axis_avg if y_axis_avg < 0 else df_y['y_axis'] < -y_axis_avg
    df_y.loc[y_condition, ['x_axis', 'y_axis', 'z_axis', 'angle']] = np.nan

    x_axis_avg = df_y['x_axis'].mean()

    df_x = df_y.copy()
    x_condition = df_x['x_axis'] > -x_axis_avg + 0.1 if x_axis_avg < 0 else df_x['x_axis'] < -x_axis_avg + 0.1
    df_x.loc[x_condition, ['x_axis', 'y_axis', 'z_axis', 'angle']] = np.nan

    # Multiple outlier removal passes
    filtered_df = remove_outliers(df_x, threshold=0.5)
    filtered_df = remove_outliers(filtered_df, threshold=0.3)
    filtered_df = remove_outliers(filtered_df, threshold=0.3)

    result_df = interpolate_axes_from_existing(filtered_df, df)

    # Apply Gaussian smoothing
    sigma = 10
    smoothed_df = result_df.copy()
    for axis in ['x_axis', 'y_axis', 'z_axis']:
        smoothed_df[axis] = gaussian_filter1d(smoothed_df[axis].interpolate(), sigma=sigma)

    # Scaling and computing z_axis
    df_scaled = scale_y_axis(scale_x_axis(smoothed_df))
    df_processed = compute_z_axis_from_xy(df_scaled, y_axis_avg)

    # Post-process angle
    df['angle'] = remove_angle_outliers(df['angle'], threshold=0.5)
    first_valid_index = df_processed['x'].first_valid_index()
    last_valid_index = df_processed['x'].last_valid_index()

    df.loc[first_valid_index:last_valid_index, 'angle'] = (
        df.loc[first_valid_index:last_valid_index, 'angle']
        .interpolate(method='linear', limit_direction='both')
    )
    for _ in range(2):
        df['angle'] = smooth_series(df['angle'].copy(), first_valid_index, last_valid_index, window=25)

    df_processed['angle'] = df['angle']
    df_processed.to_csv(output_csv_path, index=False)
    print(f"Saved rotation data to {output_csv_path}")


# ==============================================================================
#                                  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    
    # VIDEO_NUMBER = "7"
    # PROJECT_ROOT = Path().resolve().parent.parent
    # INPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "spin" / "intermediate_data" / f"Rotation_data_{VIDEO_NUMBER}.csv")
    # OUTPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "spin" / "intermediate_data" / f"Rotation_data_processed_{VIDEO_NUMBER}.csv")

    spin_detection(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
