import cv2
import numpy as np
import pandas as pd

from lane_detection.Upper_Line_Detection import get_intersection

""" Create a df of the 4 corners of the lane from the 4 lines"""


def create_points_df(bottom_lines, left_lines, right_lines, top_lines):
    columns = [
        "Frame",
        "bottom_left_x",
        "bottom_left_y",
        "bottom_right_x",
        "bottom_right_y",
        "up_left_x",
        "up_left_y",
        "up_right_x",
        "up_right_y",
    ]
    data = []
    max_iterations = min(
        len(bottom_lines), len(left_lines), len(right_lines), len(top_lines)
    )
    for i in range(max_iterations):
        bottom_left = get_intersection(bottom_lines[i], left_lines[i])
        bottom_right = get_intersection(bottom_lines[i], right_lines[i])
        up_left = get_intersection(top_lines[i], left_lines[i])
        up_right = get_intersection(top_lines[i], right_lines[i])

        data.append(
            [
                i,
                bottom_left[0],
                bottom_left[1],
                bottom_right[0],
                bottom_right[1],
                up_left[0],
                up_left[1],
                up_right[0],
                up_right[1],
            ]
        )

    # Create the DataFrame
    points_df = pd.DataFrame(data, columns=columns)

    return points_df


"""if both are above return them, otherwise None """


def is_disappeared(bl_0, br_0, bl, br, tr, tl, max_y, threshold=0.99):
    if (
        bl[1] < max_y * threshold and br[1] < max_y * threshold
    ):  # both points are above the threshold -> keep them
        return bl, br
    return None, None


""" Remove outliers and compute the missing coordinates"""


def postprocessing_upper(points_df, bottom_y_distances):
    df = points_df[["Frame", "up_left_y"]].copy()
    df_length = len(df)

    distances = np.diff(df["up_left_y"].values)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    threshold = std_distance
    num_outliers = 100

    while num_outliers > 0:
        filtered_distances = np.where(np.abs(distances) > threshold, np.nan, distances)
        num_outliers = np.sum(np.isnan(filtered_distances))

        # Find NaN indeces
        nan_indices = np.where(np.isnan(filtered_distances))[0]
        # compute next indeces (I want to remove them from df)
        next_indices = nan_indices + 1
        next_indices = next_indices[next_indices < len(df)]
        # Remove lines
        df = df.drop(index=df.index[next_indices]).reset_index(drop=True)

        # compute angain the distances
        distances = np.diff(df["up_left_y"].values)

    # interpolate to found the missing values
    df["up_left_y"] = df["up_left_y"].interpolate(
        method="linear", limit_direction="both"
    )

    # fill the remaining values at the end of the df with estimated values from the bottom line
    for i in range(len(df), df_length):
        if i >= len(df):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Frame": [i],
                            "up_left_y": [
                                df.loc[i - 1, "up_left_y"] + bottom_y_distances[i - 1]
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # update points_df with the new values
    for i, row in df.iterrows():
        y = row["up_left_y"]
        left_intersection = get_intersection(
            [
                points_df.loc[i, "bottom_left_x"],
                points_df.loc[i, "bottom_left_y"],
                points_df.loc[i, "up_left_x"],
                points_df.loc[i, "up_left_y"],
            ],
            [0, y, 1000, y],
        )
        right_intersection = get_intersection(
            [
                points_df.loc[i, "bottom_right_x"],
                points_df.loc[i, "bottom_right_y"],
                points_df.loc[i, "up_right_x"],
                points_df.loc[i, "up_right_y"],
            ],
            [0, y, 1000, y],
        )
        if left_intersection is not None and right_intersection is not None:
            points_df.loc[i, "up_left_x"] = left_intersection[0]
            points_df.loc[i, "up_left_y"] = left_intersection[1]
            points_df.loc[i, "up_right_x"] = right_intersection[0]
            points_df.loc[i, "up_right_y"] = right_intersection[1]
    return points_df


""" When the bottom line is visible it adjust the top line
when it is not visible anymore, it compute the bottom line starting from the top one"""


def postprocessing_top_bottom(points_df, cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Get the height of the frame
    frame = cap.read()[1]
    height = frame.shape[0]

    # flag for the detection of the frame where the bottom line disappear
    bottom_disappeared = False

    # create a copy of the df
    df_copy = points_df.copy()
    bottom_y_distances = np.diff(df_copy["bottom_left_y"].values)

    for i in range(1, len(points_df)):
        # select points
        bl_prev = (
            df_copy.iloc[i - 1]["bottom_left_x"],
            df_copy.iloc[i - 1]["bottom_left_y"],
        )
        br_prev = (
            df_copy.iloc[i - 1]["bottom_right_x"],
            df_copy.iloc[i - 1]["bottom_right_y"],
        )
        tl_prev = (df_copy.iloc[i - 1]["up_left_x"], df_copy.iloc[i - 1]["up_left_y"])
        tr_prev = (df_copy.iloc[i - 1]["up_right_x"], df_copy.iloc[i - 1]["up_right_y"])
        bl = (points_df.iloc[i]["bottom_left_x"], points_df.iloc[i]["bottom_left_y"])
        br = (points_df.iloc[i]["bottom_right_x"], points_df.iloc[i]["bottom_right_y"])
        tr = (points_df.iloc[i]["up_right_x"], points_df.iloc[i]["up_right_y"])
        tl = (points_df.iloc[i]["up_left_x"], points_df.iloc[i]["up_left_y"])

        # case 1: the bottom line is visible
        if not bottom_disappeared:
            # select data in a sliding window
            window_size = 9
            if i < window_size:
                left_relative_position = (
                    tl_prev[0] - bl_prev[0],
                    tl_prev[1] - bl_prev[1],
                )
                right_relative_position = (
                    tr_prev[0] - br_prev[0],
                    tr_prev[1] - br_prev[1],
                )
            else:
                left_relative_positions = [
                    (
                        df_copy.iloc[j]["up_left_x"] - df_copy.iloc[j]["bottom_left_x"],
                        df_copy.iloc[j]["up_left_y"] - df_copy.iloc[j]["bottom_left_y"],
                    )
                    for j in range(i - window_size + 1, i + 1)
                ]
                right_relative_positions = [
                    (
                        df_copy.iloc[j]["up_right_x"]
                        - df_copy.iloc[j]["bottom_right_x"],
                        df_copy.iloc[j]["up_right_y"]
                        - df_copy.iloc[j]["bottom_right_y"],
                    )
                    for j in range(i - window_size + 1, i + 1)
                ]

                # select the second lower line in the frame
                left_relative_position = sorted(
                    left_relative_positions, key=lambda pos: pos[1], reverse=True
                )[2]
                right_relative_position = sorted(
                    right_relative_positions, key=lambda pos: pos[1], reverse=True
                )[2]

            # compute the new position of the bottom points in the current frame (if needed)
            bl_new, br_new = is_disappeared(bl_prev, br_prev, bl, br, tr, tl, height)

            if bl_new is None and br_new is None:
                bottom_disappeared = True
                index_disappeared = i
                print(f"Bottom points disappeared at frame {i}.")
            else:  # consider correct the bottom points
                tr_mid = (
                    br_new[0] + right_relative_position[0],
                    br_new[1] + right_relative_position[1],
                )
                tl_mid = (
                    bl_new[0] + left_relative_position[0],
                    bl_new[1] + left_relative_position[1],
                )
                # Calculate the intersection point
                tr_new = get_intersection(
                    [tr_mid[0], tr_mid[1], tl_mid[0], tl_mid[1]],
                    [br_new[0], br_new[1], tr[0], tr[1]],
                )
                tl_new = get_intersection(
                    [tl_mid[0], tl_mid[1], tr_mid[0], tr_mid[1]],
                    [bl_new[0], bl_new[1], tl[0], tl[1]],
                )

        if bottom_disappeared:  # consider correct the top poits
            bl_new = (
                tl[0] - left_relative_position[0],
                tl[1] - left_relative_position[1],
            )
            br_new = (
                tr[0] - right_relative_position[0],
                tr[1] - right_relative_position[1],
            )
            tr_new = tr
            tl_new = tl

        # Update the DataFrame with the new points
        points_df.at[i, "bottom_left_x"] = bl_new[0]
        points_df.at[i, "bottom_left_y"] = bl_new[1]
        points_df.at[i, "bottom_right_x"] = br_new[0]
        points_df.at[i, "bottom_right_y"] = br_new[1]
        points_df.at[i, "up_left_x"] = tl_new[0]
        points_df.at[i, "up_left_y"] = tl_new[1]
        points_df.at[i, "up_right_x"] = tr_new[0]
        points_df.at[i, "up_right_y"] = tr_new[1]

    # Postprocessingg of the upper lines to remove outliers
    df = postprocessing_upper(points_df, bottom_y_distances)
    # recompute the bottom points based on the new upper line (if the bottom line is not visible anymore)
    if bottom_disappeared:
        for i in range(index_disappeared, len(points_df)):
            # compute the bottom points in the rows that are changed between df and points_df with relative positions
            bl_new = (
                df.iloc[i]["up_left_x"] - left_relative_position[0],
                df.iloc[i]["up_left_y"] - left_relative_position[1],
            )
            br_new = (
                df.iloc[i]["up_right_x"] - right_relative_position[0],
                df.iloc[i]["up_right_y"] - right_relative_position[1],
            )

            # update df with the new points
            df.at[i, "bottom_left_x"] = bl_new[0]
            df.at[i, "bottom_left_y"] = bl_new[1]
            df.at[i, "bottom_right_x"] = br_new[0]
            df.at[i, "bottom_right_y"] = br_new[1]

    return df


def postprocessing_top_still(points_df):
    # Create a copy of the DataFrame to avoid modifying the original one
    df_copy = points_df.copy()

    # Compute the mean of the 'up_left_y' column for the first half of the DataFrame
    mean_value = df_copy.iloc[: len(df_copy) // 2]["up_left_y"].mean()

    # Compute the intersection between the horizontal line with y coordinate equal to the mean value and the laetral lines
    for i in range(len(df_copy)):
        # Compute the intersection points
        left_intersection = get_intersection(
            [
                points_df.iloc[i]["bottom_left_x"],
                points_df.iloc[i]["bottom_left_y"],
                points_df.iloc[i]["up_left_x"],
                points_df.iloc[i]["up_left_y"],
            ],
            [0, mean_value, 1000, mean_value],
        )
        right_intersection = get_intersection(
            [
                points_df.iloc[i]["bottom_right_x"],
                points_df.iloc[i]["bottom_right_y"],
                points_df.iloc[i]["up_right_x"],
                points_df.iloc[i]["up_right_y"],
            ],
            [0, mean_value, 1000, mean_value],
        )
        if left_intersection is not None and right_intersection is not None:
            df_copy.at[i, "up_left_x"] = left_intersection[0]
            df_copy.at[i, "up_left_y"] = left_intersection[1]
            df_copy.at[i, "up_right_x"] = right_intersection[0]
            df_copy.at[i, "up_right_y"] = right_intersection[1]

    return df_copy


def publish_csv_lane_points(output_path, output_df):
    # Save the DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)
    print(f"CSV file with the lane points saved to {output_path}")

    return


def postprocessing_upper_lines(
    cap, bottom_lines, left_lines, right_lines, upper_lines_raw, avg_motion
):
    # Create a DataFrame with the lane points
    points_df = create_points_df(bottom_lines, left_lines, right_lines, upper_lines_raw)

    # Distinguish between the two cases: when the video is still and when it is moving
    if avg_motion > 1:
        # postprocessing teh upper lines and adjusting the bottom ones when they disappear
        points_df = postprocessing_top_bottom(points_df, cap)
    else:
        # postprocessing the upper lines when the video is still
        points_df = postprocessing_top_still(points_df)

    return points_df


""" draw one line on the frame"""


def draw_line_on_frame(frame, line):
    # Create a copy of the original frame to draw the first line
    modified_frame = np.copy(frame)

    # Extract the first line's rho and theta
    if line is not None:
        x1, y1, x2, y2 = line
        # Extend the line to the image boundaries
        # [x1_ext, y1_ext, x2_ext, y2_ext] = get_extended_line(line, frame.shape[1], frame.shape[0])

        # Draw the first line on the frame
        cv2.line(modified_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # return the modified frame
    return modified_frame


""" draw the lines on the frame"""


def draw_lines_on_frame(frame, lines):
    for i in range(len(lines)):
        frame = draw_line_on_frame(frame, lines[i])
    return frame


""" Generate the video with the lines"""


def generate_video_lines(cap, output_path, points_df):
    # start the video from the begnning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' codec for MP4 format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # Loop through each frame in the video
    frame_index = 0
    while frame_index < len(points_df):
        ret, video_frame = cap.read()
        if not ret:
            break

        # draw the lines on the frame
        lines = [
            [
                points_df.iloc[frame_index]["bottom_left_x"],
                points_df.iloc[frame_index]["bottom_left_y"],
                points_df.iloc[frame_index]["bottom_right_x"],
                points_df.iloc[frame_index]["bottom_right_y"],
            ],
            [
                points_df.iloc[frame_index]["up_left_x"],
                points_df.iloc[frame_index]["up_left_y"],
                points_df.iloc[frame_index]["up_right_x"],
                points_df.iloc[frame_index]["up_right_y"],
            ],
            [
                points_df.iloc[frame_index]["bottom_left_x"],
                points_df.iloc[frame_index]["bottom_left_y"],
                points_df.iloc[frame_index]["up_left_x"],
                points_df.iloc[frame_index]["up_left_y"],
            ],
            [
                points_df.iloc[frame_index]["bottom_right_x"],
                points_df.iloc[frame_index]["bottom_right_y"],
                points_df.iloc[frame_index]["up_right_x"],
                points_df.iloc[frame_index]["up_right_y"],
            ],
        ]
        modified_frame = draw_lines_on_frame(video_frame, lines)

        # Write the modified frame to the output video
        out.write(modified_frame)

        # Increment the frame index
        frame_index += 1

    # Release the video capture and writer objects
    out.release()
    cap.release()

    print(f"Video with three lines saved to {output_path}")
