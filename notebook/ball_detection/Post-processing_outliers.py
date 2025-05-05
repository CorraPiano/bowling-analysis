import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression


# ==============================================================================
#                              AUXILIARY FUNCTIONS
# ==============================================================================

def decreasing_func(x, a, b, c):
    """
    Exponential decay function to model the radius over time.
    Returns:
    array-like
        The computed radius values.
    """
    return a * np.exp(-b * x) + c


def process_radius(df, median_window=25, quantile_baseline=0.05, ransac_threshold=0.5):
    """
    Process the radius data in the DataFrame using a robust fitting method.
    Returns:
    pd.DataFrame
        The processed DataFrame with fitted radius values.
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    all_frames = pd.DataFrame({'frame': np.arange(int(df['frame'].min()), int(df['frame'].max()) + 1)})
    df = pd.merge(all_frames, df, on='frame', how='left')

    df['median'] = df['radius'].rolling(window=median_window, center=True, min_periods=1).median()

    c_est = df['radius'].quantile(quantile_baseline)

    eps = 1e-6
    valid = df['radius'] > (c_est + eps)

    if valid.sum() < 10:
        raise ValueError("Too few valid points for robust fitting. Consider adjusting parameters.")

    x = df.loc[valid, 'frame'].values.reshape(-1, 1)
    y = np.log(df.loc[valid, 'radius'].values - c_est)

    ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=ransac_threshold, max_trials=1000)
    ransac.fit(x, y)

    slope = -ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    a_est = np.exp(intercept)

    full_x = df['frame'].values
    fitted = decreasing_func(full_x, a_est, slope, c_est)
    df['radius'] = np.round(fitted).astype(int)

    return df


def delete_outliers(df, processed_df):
    """
    Delete outliers based on the processed radius values.
    Returns:
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    threshold = 10
    outlier_mask = (df['radius'] > processed_df['radius'] + threshold) | (df['radius'] < processed_df['radius'] - threshold)
    cleaned_df = df.copy()
    cleaned_df.loc[outlier_mask, ['radius', 'x', 'y']] = np.nan

    return cleaned_df

# ==============================================================================
#                              RADIUS PROCESSING
# ==============================================================================


def process_data(input_csv: str, output_csv: str):
    """
    Process the radius data from the input CSV file and save the cleaned data to the output CSV file.
    """
    df = pd.read_csv(input_csv)
    processed_df = process_radius(df, median_window=21, quantile_baseline=0.05, ransac_threshold=0.4)
    df_cleaned = delete_outliers(df, processed_df)
    df_cleaned.to_csv(output_csv, index=False)

    print("Processed DataFrame saved to: ", output_csv)


# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    #PROJECT_ROOT = Path().resolve().parent.parent
    #INPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "ball_detection" / "intermediate_data" / f"Circle_positions_raw_{VIDEO_NUM}.csv")
    #OUTPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "ball_detection" / "intermediate_data" / f"Circle_positions_cleaned_{VIDEO_NUM}.csv")

    process_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
