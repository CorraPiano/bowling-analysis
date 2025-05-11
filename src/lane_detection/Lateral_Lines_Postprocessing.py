import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from lane_detection.Bottom_Line_Detection import calculate_angle
from lane_detection.Bottom_Line_Postprocessing import get_relevant_points, interpolate_missing_coordinates, points_to_lines

''' execute an iterated algorithm for removing outliers,
    the exit threshold is set as two times the mean of the distances between points at the first iteration'''
def remove_outliers_iterated(df, threshold_factor=1):
    # Ensure 'X' and 'Y' are numeric
    df[['X', 'Y']] = df[['X', 'Y']].apply(pd.to_numeric, errors='coerce')

    first_iteration = True
    exit_threshold = 10000
    max_dist = np.inf

    # Compute the distance between consecutive points
    dx = df['X'].diff()
    dy = df['Y'].diff()

    # compute distances
    df['distance'] = np.sqrt(dx**2 + dy**2)

    # remove 0,0 points
    df = df[(df['X'] != 0) | (df['Y'] != 0)]
    
    while max_dist > exit_threshold:

        # set the mean to exit the while
        if first_iteration:
            first_iteration = False
            mean_dist = np.mean(df['distance'])
            std = np.std(df['distance'])
            if std > mean_dist:
                exit_threshold = threshold_factor * std
            else:
                exit_threshold = threshold_factor * 2 * mean_dist

        # set the threshold
        threshold_dist = 2 * np.mean(df['distance'])
        # remove outliers
        df = df[(df['distance'] < threshold_dist) & df['distance'].notna()]  # the first value is considered as correct

        # Compute the distance between consecutive points
        dx = df['X'].diff()
        dy = df['Y'].diff()

        # compute distances
        df['distance'] = np.sqrt(dx**2 + dy**2)

        # get the max distance
        max_dist = np.max(df['distance'])

    return df

''' Remove the lines with a slope too different from the average one'''
def remove_outliers_slope(lines_df):
    # add a column to df with the angle computed with calculate_angle
    lines_df['slope'] = lines_df.apply(
        lambda row: calculate_angle(row['X1'], row['Y1'], row['X2'], row['Y2']), axis=1
    )
    # compute average and stdd
    mean_angle = lines_df['slope'].mean()
    std_angle = lines_df['slope'].std()

    # Compute the tolerance
    tolerance = std_angle

    lines_df.loc[
        (lines_df['slope'] < mean_angle - tolerance) | (lines_df['slope'] > mean_angle + tolerance),
        ['X1', 'Y1', 'X2', 'Y2']
    ] = 0

    filtered_lines = lines_df[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

    return filtered_lines

''' Remove the outliers with DBSCAN'''
def remove_outliers_dbscan(df):
    # remove the points that are 0,0
    df = df[(df['X'] != 0) | (df['Y'] != 0)]

    # df deve avere colonne 'X' e 'Y'
    X = df[['X', 'Y']].values

    # Apply dbscan
    db = DBSCAN(eps=30, min_samples=5).fit(X)

    df['label'] = db.labels_

    # Rimuove i punti classificati come noise (-1)
    df_clean = df[df['label'] != -1]
    
    return df_clean

''' Remove the outliers separately for x and y'''
def remove_outliers_xy(df):
    # compute the mean and std of the X and Y coordinates
    mean_x = df['X'].mean()
    std_x = df['X'].std()
    mean_y = df['Y'].mean()
    std_y = df['Y'].std()

    # set the threshold
    threshold_x =  std_x * 2
    threshold_y =  std_y * 2
    # remove outliers: remove the points that have a distance on each axis with respect to the previous point greater than the threshold
    mask = []
    for i in range(len(df.index.values)):
        if df['X'].iloc[i] == 0 and df['Y'].iloc[i] == 0:
            mask.append(df.index.values[i])
            continue
        # compute the first feasible previous index
        prev_index = i - 1
        while df.index.values[prev_index] in mask and prev_index >= 0:
            prev_index -= 1
        if (abs(df['X'].iloc[i] - df['X'].iloc[prev_index]) > threshold_x or abs(df['Y'].iloc[i] - df['Y'].iloc[prev_index]) > threshold_y) and prev_index >= 0:
            mask.append(df.index.values[i])
    
    # Remove the outlier points from the DataFrame
    df = df.drop(mask)

    return df

''' Postprocessing of one of the 2 lateral lines'''
def postprocessing_lines_iterated(lines, avg_movement, threshold_factor=1):
    # Se lines è una Series o lista di array/liste
    lines_with_frames = [
        [frame_idx] + list(line) for frame_idx, line in enumerate(lines)
    ]

    # Create the line df
    lines_df = pd.DataFrame(lines_with_frames, columns=['Frame', 'X1', 'Y1', 'X2', 'Y2'])
    # Remove teh outliers based on the slope
    lines_cleaned = remove_outliers_slope(lines_df)

    # find the point of the lines closet to the origin and create the proper dataframe
    closest_points = get_relevant_points(lines_cleaned)
   
    # create a proper DataFrame
    closest_points_df = pd.DataFrame(closest_points, columns=['X', 'Y'])
    closest_points_df['Frame'] = range(len(closest_points_df))
    closest_points_df = closest_points_df[['Frame', 'X', 'Y']]

    if avg_movement > 1:
         # Remove outliers with DBSCAN
        points_cleaned_dbscan = remove_outliers_dbscan(closest_points_df)

        # Remove outliers with a distance threshold
        points_cleaned = remove_outliers_xy(points_cleaned_dbscan)

        # Interpolate missing coordinates
        points_interpolated = interpolate_missing_coordinates(points_cleaned, len(lines)-1)

        processed_line = points_to_lines(points_interpolated)
    else:
        # Remove outliers iterated
        points_cleaned = remove_outliers_iterated(closest_points_df, threshold_factor=avg_movement)

        #get the mean point because I'm assuming the video still
        mean_point = points_cleaned.mean()

        #create a list of length of the original df where each element is mean_point
        mean_point_list = [mean_point] * len(closest_points_df)
        mean_point_df = pd.DataFrame(mean_point_list, columns=['X', 'Y'])

        processed_line = points_to_lines(mean_point_df)

    return processed_line 

''' Postprocessing of the lateral lines, done separately'''
def postprocessing_lateral_lines(left_lines, right_lines, avg_movement):
    left_lines_processed = postprocessing_lines_iterated(left_lines, avg_movement)
    right_lines_processed = postprocessing_lines_iterated(right_lines, avg_movement)
    return left_lines_processed, right_lines_processed


