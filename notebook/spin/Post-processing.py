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

def interpolate_axes_from_b(A, B):

    # Ensure 'frame' column exists in both
    if 'frame' not in A.columns or 'frame' not in B.columns:
        raise ValueError("Both CSVs must contain a 'frame' column.")

    # Identify frames in B where x and y are not NaN
    valid_frames = B.dropna(subset=['x', 'y'])['frame'].unique()

    # Make a copy to avoid modifying the original
    A_interp = A.copy()

    # Create a mask for valid frames in A
    valid_mask = A_interp['frame'].isin(valid_frames)

    for axis in ['x_axis', 'y_axis', 'z_axis']:
        # Work only with rows in A that correspond to valid frames
        axis_series = A_interp.loc[valid_mask, axis]

        # Interpolate based only on this subset
        interpolated_values = axis_series.interpolate(method='linear', limit_direction='both')

        # Assign the interpolated values back
        A_interp.loc[valid_mask, axis] = interpolated_values

    return A_interp

def enforce_non_decreasing_x_axis(df):
    if 'x_axis' not in df.columns:
        raise ValueError("The DataFrame must contain a column named 'x_axis'")

    last_valid = df.loc[0, 'x_axis']
    for i in range(1, len(df)):
        if df.loc[i, 'x_axis'] < last_valid:
            df.loc[i, 'x_axis'] = last_valid
        else:
            last_valid = df.loc[i, 'x_axis']
    
    return df


# ==============================================================================
#                             SCALING AND SMOOTHING
# ==============================================================================

def scale_x_axis(df):
    df = df.copy()

    # Get index and values of first and last valid x_axis
    valid_x = df['x_axis'].dropna()
    if valid_x.empty:
        return df  # nothing to scale

    first_idx = valid_x.index[0]
    last_idx = valid_x.index[-1]
    x_start = valid_x.iloc[0]
    x_end = 1 / valid_x.iloc[-1]
    num_rows = last_idx - first_idx + 1

    # Generate scale factors
    scale_factors = np.linspace(x_start, x_end, num_rows)

    # Apply scaling
    df.loc[first_idx:last_idx, 'x_axis'] = df.loc[first_idx:last_idx, 'x_axis'] * scale_factors

    return df


def scale_y_axis(df):
    df = df.copy()

    # Get index and values of first and last valid y_axis
    valid_y = df['y_axis'].dropna()
    if valid_y.empty:
        return df  # nothing to scale

    first_idx = valid_y.index[0]
    last_idx = valid_y.index[-1]
    y_start = 1
    y_end = 0
    num_rows = last_idx - first_idx + 1

    # Generate scale factors from y_start to y_end
    scale_factors = np.linspace(y_start, y_end, num_rows)

    # Apply scaling
    df.loc[first_idx:last_idx, 'y_axis'] = df.loc[first_idx:last_idx, 'y_axis'] * scale_factors

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
    
    # Compute 1 - x^2 - y^2
    squared_sum = df['x_axis']**2 + df['y_axis']**2
    z_values = 1 - squared_sum

    # Handle invalid values (e.g., where 1 - x^2 - y^2 < 0)
    z_values[z_values < 0] = np.nan

    # Calculate z = sqrt(1 - x^2 - y^2)
    if y_axis_avg < 0:
        df['z_axis'] = np.sqrt(z_values)
    else:
        df['z_axis'] = -np.sqrt(z_values)

    return df


# ==============================================================================
#                             MAIN VIDEO PROCESSING
# ==============================================================================

def spin_detection(input_video_path, output_csv_path, input_original_csv_path):

    # TODO: apply fill_frames

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

    df_original = pd.read_csv(input_original_csv_path)
    result_df = interpolate_axes_from_b(filtered_df, df_original)

    # Apply Gaussian smoothing
    sigma = 10
    smoothed_df = result_df.copy()

    for axis in ['x_axis', 'y_axis', 'z_axis']:
        original = result_df[axis]
        smoothed = original.copy()

        # Find contiguous non-NaN segments
        not_nan = original.notna()
        group = (not_nan != not_nan.shift()).cumsum()
        
        for grp_id, is_valid in group[not_nan].groupby(group):
            segment_indices = is_valid.index
            segment_values = original.loc[segment_indices]

            # Interpolate within segment (optional if there are still NaNs)
            interpolated = segment_values.interpolate()
            # Apply smoothing only to that segment
            smoothed_segment = gaussian_filter1d(interpolated, sigma=sigma, mode='nearest')

            # Assign back to output
            smoothed.loc[segment_indices] = smoothed_segment

        # Store in result
        smoothed_df[axis] = smoothed
    
    smoothed_df = enforce_non_decreasing_x_axis(smoothed_df)

    # Scaling and computing z_axis
    df_scaled = scale_x_axis(smoothed_df)
    df_scaled = scale_y_axis(df_scaled)
    df_processed = compute_z_axis_from_xy(df_scaled, y_axis_avg)

    # Create a mask for rows where A has NaN in all three columns and B has valid values
    mask = (
        df_processed['x'].isna() & df_processed['y'].isna() & df_processed['radius'].isna() &
        df_original['x'].notna() & df_original['y'].notna() & df_original['radius'].notna()
    )

    # Update A with values from B where the mask is True
    df_processed.loc[mask, ['x', 'y', 'radius']] = df_original.loc[mask, ['x', 'y', 'radius']]

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

    spin_detection(INPUT_CSV_PATH, OUTPUT_CSV_PATH, INPUT_ORIGINAL_CSV_PATH)
