import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import medfilt


''' get the intersection of each line with the orthogonal line passing through the origin'''
def get_relevant_points(lines):
    
    # initialize the list of intersection points
    intersection_points = []
    for line in lines:
        if line is not None:
            x1, y1, x2, y2 = line
            # Calculate the line coefficients (a, b, c) for ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2

            # Calculate the intersection with the orthogonal line passing through the origin
            # The orthogonal line has equation bx - ay = 0
            denominator = a**2 + b**2
            if denominator == 0:
                intersection_points.append((0, 0))
                continue  # Skip degenerate lines

            x = -a * c / denominator
            y = -b * c / denominator

            # Append the intersection point
            intersection_points.append((x, y))
        else:
            # Append a default point if the line is None
            intersection_points.append((0, 0))
    
    return intersection_points

''' Removes the outliers from the bottom line detection,
    The outliers are defined as the points which distance is too far from the std of the distances'''
def remove_outliers_bottom(df, threshold=1):
    # Ensure 'X' and 'Y' are numeric
    df[['X', 'Y']] = df[['X', 'Y']].apply(pd.to_numeric, errors='coerce')
    # Compute the distance between consecutive points
    dx = df['X'].diff()
    dy = df['Y'].diff()
    distances = np.sqrt(dx**2 + dy**2)

    # Define outlier threshold
    median_dist = distances.mean()
    std_dist = distances.std()

    outlier_threshold =  threshold * std_dist * 2

    mask = []
    
    for i in range(0, len(distances)-1):

        # Remove points that are  (0, 0)  (the ones that I set when it doesn't detect any line)
        if df['X'].iloc[i] == 0 and df['Y'].iloc[i] == 0:
            mask.append(i)
            prev_index = i - 1
            while prev_index in mask and prev_index >= 0:
                prev_index -= 1
            if prev_index < 0:
                print("No valid previous index found. Breaking. (remove_outliers_bottom)")
                break
            distances[i+1] = np.sqrt((df['X'].iloc[i+1] - df['X'].iloc[prev_index])**2 + (df['Y'].iloc[i+1] - df['Y'].iloc[prev_index])**2)
            continue
        if distances[i] > outlier_threshold:
            # Create a mask to identify the outlier point
            mask.append(i)
            # convert the next point distance in the distance with respect to the previous point
            if i < len(distances) - 1 and i > 0:
                # select the previous point as the last point evaluated non present in the mask
                prev_index = i - 1
                while prev_index in mask and prev_index >= 0:
                    prev_index -= 1
                if prev_index < 0:
                    print("No valid previous index found. Breaking.")
                    break
                distances[i+1] = np.sqrt((df['X'].iloc[i+1] - df['X'].iloc[prev_index])**2 + (df['Y'].iloc[i+1] - df['Y'].iloc[prev_index])**2)
   # Remove the outlier points from the DataFrame
    df = df.drop(mask)

    # compute means and standard deviations of points
    mean_x = df['X'].mean()
    mean_y = df['Y'].mean()
    std_x = df['X'].std()
    std_y = df['Y'].std()   


    threshold_std = 1
    if std_x > 1 and std_y > 1:
        # Remove points too far from the mean
        df = df[(df['X'] > mean_x - threshold_std*std_x) & (df['X'] < mean_x + threshold_std*std_x) & (df['Y'] > mean_y - threshold_std*std_y) & (df['Y'] < mean_y + threshold_std*std_y)]
    elif std_x > 1:
        df = df[(df['X'] > mean_x - threshold_std*std_x) & (df['X'] < mean_x + threshold_std*std_x)]
    elif std_y > 1:
        df = df[(df['Y'] > mean_y - threshold_std*std_y) & (df['Y'] < mean_y + threshold_std*std_y)]
    
    return df

''' Smooths the X and Y coordinates using a median filter to reduce noise.'''
def median_filter(df, kernel_size=3):
    df['X'] = medfilt(df['X'], kernel_size=kernel_size)
    df['Y'] = medfilt(df['Y'], kernel_size=kernel_size)
    return df

''' Smooths the X and Y coordinates using Savitzky-Golay filter to reduce noise.'''
def Savitzky_Golay_filter(df, window_length=25, polyorder=1):

    df['X'] = savgol_filter(df['X'], window_length=window_length, polyorder=polyorder)
    df['Y'] = savgol_filter(df['Y'], window_length=window_length, polyorder=polyorder)
    
    return df

'''Interpolate the missing coordinates between the frames from start_frame to end_frame.
    Ensures a smooth trajectory by interpolating both X and Y coordinates.'''
def interpolate_missing_coordinates(df, end_frame=100, start_frame=0, polyorder=1): 
    # Create a DataFrame with all frames from start_frame to end_frame
    all_frames = pd.DataFrame({'Frame': range(start_frame, end_frame + 1)})
    
    # Merge with the original dataframe
    df_full = pd.merge(all_frames, df, on='Frame', how='left')
    
    # Interpolate the missing 'X' and 'Y' coordinates (linear interpolation)
    df_full['X'] = df_full['X'].interpolate(method='linear')
    df_full['Y'] = df_full['Y'].interpolate(method='linear')

    # Fill any remaining NaN values using forward and backward filling
    df_full['X'] = df_full['X'].bfill().ffill()
    df_full['Y'] = df_full['Y'].bfill().ffill()
    
    # Apply smoothing to ensure a smooth trajectory
    df_full = Savitzky_Golay_filter(df_full, window_length=5, polyorder=polyorder)
    
    return df_full

''' Get the lines from the intersection points'''
def points_to_lines(df, line_length=10):
    lines = []
    for i in range(0, len(df)):
        x, y = df.iloc[i]['X'], df.iloc[i]['Y']
        # Compute the direction vector orthogonal to the line connecting the point to the origin
        dx, dy = -y, x  # Rotate the vector (x, y) by 90 degrees to get orthogonal direction
        # Append the line as a pair of points
        lines.append([x + line_length * dx, y + line_length * dy, x - line_length * dx, y - line_length * dy])
    return lines

''' Postprocessing of the bottom lines '''
def postprocessing_bottom_lines(horizontal_lines: list, avg_movement) -> list:
    # get the intersection points
    intersection_points = get_relevant_points(horizontal_lines)

    # Create a DataFrame from intersection_points
    points_df = pd.DataFrame(intersection_points, columns=['X', 'Y'])
    points_df['Frame'] = range(len(points_df))
    
    # Reorder columns to have 'Frame', 'X', 'Y' 
    points_df = points_df[['Frame', 'X', 'Y']]

    if avg_movement > 1:
        # Remove outliers
        points_cleaned = remove_outliers_bottom(points_df, threshold=0.5)
        # Smooth the trajectory to reduce noise
        points_smoothed = median_filter(points_cleaned)

        # Interpolate missing coordinates
        points_interpolated = interpolate_missing_coordinates(points_smoothed, len(points_df)-1)
        
        bottom_lines = points_to_lines(points_interpolated)
    else:
        # Remove outliers
        points_cleaned = remove_outliers_bottom(points_df)

        #get the mean point because I'm assuming the video still
        mean_point = points_cleaned.mean()
        # print('mean point:', mean_point)

        #create a list of length of the original df where each element is mean_point
        mean_point_list = [mean_point] * len(intersection_points)
        mean_point_df = pd.DataFrame(mean_point_list, columns=['X', 'Y'])

        bottom_lines = points_to_lines(mean_point_df)

    return bottom_lines

