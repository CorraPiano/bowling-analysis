import cv2
import numpy as np
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

    all_frames = pd.DataFrame(
        {"frame": np.arange(int(df["frame"].min()), int(df["frame"].max()) + 1)}
    )
    df = pd.merge(all_frames, df, on="frame", how="left")

    df["median"] = (
        df["radius"].rolling(window=median_window, center=True, min_periods=1).median()
    )

    c_est = df["radius"].quantile(quantile_baseline)

    eps = 1e-6
    valid = df["radius"] > (c_est + eps)

    if valid.sum() < 10:
        raise ValueError(
            "Too few valid points for robust fitting. Consider adjusting parameters."
        )

    x = df.loc[valid, "frame"].values.reshape(-1, 1)
    y = np.log(df.loc[valid, "radius"].values - c_est)

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=ransac_threshold,
        max_trials=1000,
    )
    ransac.fit(x, y)

    slope = -ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    a_est = np.exp(intercept)

    full_x = df["frame"].values
    fitted = decreasing_func(full_x, a_est, slope, c_est)
    df["radius"] = np.round(fitted).astype(int)

    return df


def process_video(input_video, output_video, df_adjusted):
    """
    Process the video to draw circles on the detected positions.
    """
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in df_adjusted.index:
            row = df_adjusted.loc[frame_number]
            x, y, radius = int(row["x"]), int(row["y"]), int(row["radius"])

            cv2.circle(frame, (x, y), radius, color=(0, 255, 0), thickness=2)
            cv2.circle(frame, (x, y), 5, color=(0, 0, 255), thickness=-1)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    print(f"Output video saved at: {output_video}")


# ==============================================================================
#                              RADIUS PROCESSING
# ==============================================================================


def process_coordinates_final(
    input_video, input_csv, transformed_csv, output_csv, output_video
):
    """
    Process the radius data from the input CSV file and save the cleaned data to the output CSV file.
    """
    df_transformed = pd.read_csv(transformed_csv)
    df_original = pd.read_csv(input_csv)

    df_merged = pd.merge(
        df_transformed, df_original[["frame", "radius"]], on="frame", how="left"
    )

    processed_df = process_radius(
        df_merged, median_window=5, quantile_baseline=0.001, ransac_threshold=0.1
    )

    processed_df["y"] = processed_df.apply(
        lambda row: row["y"] - row["radius"]
        if pd.notna(row["y"]) and pd.notna(row["radius"])
        else row["y"],
        axis=1,
    )

    for col in ["x", "y", "radius"]:
        processed_df[col] = (
            pd.to_numeric(processed_df[col], errors="coerce").dropna().astype(int)
        )

    df_output = processed_df[["frame", "x", "y", "radius"]]
    df_output.to_csv(output_csv, index=False)

    print(f"Adjusted positions saved to: {output_csv}")

    df_adjusted = pd.read_csv(output_csv).set_index("frame")
    process_video(input_video, output_video, df_adjusted)
