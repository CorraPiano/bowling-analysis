import cv2
from pathlib import Path
import numpy as np
import math
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from sklearn.cluster import DBSCAN
import time


# ==============================================================================
#                              BACKGROUND MOTION FUNCTIONS
# ==============================================================================
def estimate_background_motion(cap: cv2.VideoCapture) -> float:
    orb = cv2.ORB_create(nfeatures=1000)  # more features helps
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prev_frame = cap.read()
    if not ret:
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)

    motions = []
    dxs = []
    dys = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)

        if prev_desc is not None and desc is not None:
            # Match ORB descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_desc, desc)

            # Extract matched keypoints
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Use RANSAC to filter out moving objects
            if len(src_pts) >= 10:   # tune
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    # Extract translation components from homography
                    dx, dy = H[0, 2], H[1, 2]
                    motion_magnitude = np.sqrt(dx**2 + dy**2)
                    motions.append(motion_magnitude)
                    dxs.append(dx)
                    dys.append(dy)

        prev_gray = gray
        prev_kp, prev_desc = kp, desc

    avg_motion = np.mean(motions)
    print("Average motion:", avg_motion)
    return avg_motion

# ==============================================================================
#                              BOTTOM LINE DETECTION FUNCTIONS
# ==============================================================================

''' get the edges from the frame (extract only brown and rose) using Canny and with the otsu threshold'''
def get_edges(frame, blur = False):
    # Define the range for light brown color in HSV
    lower_brown = np.array([00, 30, 100])
    upper_brown = np.array([20, 200, 255])

    # Define the range for rose color in HSV
    lower_rose = np.array([150, 30, 200])
    upper_rose = np.array([180, 200, 255])
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for brown and rose colors
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
    mask_rose = cv2.inRange(hsv_image, lower_rose, upper_rose)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_brown, mask_rose)

    # apply brown and rose mask
    extracted_image = cv2.bitwise_and(frame, frame, mask=combined_mask)

    if blur == True:
        # blur the image
        extracted_image = cv2.GaussianBlur(extracted_image, (15, 15), 0)


    # Convert the bottom image to grayscale
    gray_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)

    # Compute Otsu's threshold 
    otsu_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Set lower and upper thresholds relative to Otsu's threshold
    lower = 0.5 * otsu_thresh
    upper = 1.5 * otsu_thresh

    # get edges
    edges = cv2.Canny(gray_image, lower, upper)

    return edges

''' Get the lines from the edges using probabilistic hough transfor'''
def get_lines_pht(edges, min_line_length, max_line_gap):    
    lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines_p

''' Compute the slope of the line'''
def calculate_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

''' Filter only the lines that are 'quite horizontal' '''
def get_horizontal(lines_p, tolerance = 5):
    horizontal = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            angle = calculate_angle(x1, y1, x2, y2)
            if abs(angle) <= tolerance:
                horizontal.append(line)
    return horizontal

''' Detection of the bottom line of the frame'''
def bottom_detection(frame):
    # Crop the bottom part of the image
    limit_y = math.floor(3/4*frame.shape[0])
    frame_bottom = frame[limit_y:frame.shape[0], 0:frame.shape[1]]

    # get edges
    edges = get_edges(frame_bottom)

    # parameters to set in PHoughTransform
    min_line_length = 50
    max_line_gap = 10 
    # get the lines
    lines_p = get_lines_pht(edges, min_line_length, max_line_gap)

    # filter horizontal lines
    horizontal_lines = get_horizontal(lines_p)

    if len(horizontal_lines) == 0:
        return None
    
    # get the first line in the list (best one)
    horizontal_line = horizontal_lines[0][0]
    # adjust y coordinates to come back to the original image points
    horizontal_line[1] = horizontal_line[1] + limit_y
    horizontal_line[3] = horizontal_line[3] + limit_y

    # return the horizontal line
    return horizontal_line

'''Get the bottom lines of the video'''
def get_bottom_lines(cap) -> list:
    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize num_frame e horizontal_line
    horizontal_lines = [None] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frame = 0

    # Loop through each frame in the video
    while num_frame < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the frame (bottom detection):", num_frame)
            break
        
        # Perform operations on the current frame
        horizontal_lines[num_frame] = bottom_detection(frame)

        # Increment the frame counter
        num_frame += 1

    return horizontal_lines

# ==============================================================================
#                            POST PROCESSING BOTTOM LINE FUNCTIONS
# ==============================================================================

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

# ==============================================================================
#                              LATERAL LINE DETECTION FUNCTIONS
# ==============================================================================

''' select the point at the center (x) of the frame on the line'''
def select_central_point(line, frame):
    x1, y1, x2, y2 = line

    # Get the x-size of the image
    x_size = frame.shape[1]
    x_half = x_size // 2

    if x2 - x1 == 0:
        # Vertical line: return the point directly
        return x1, (y1 + y2) // 2

    # Calculate slope (m) and intercept (q) of the line
    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1

    # Calculate the y value at x = x_half
    y_half = int(m * x_half + q)

    return x_half, y_half

''' transform the line composed by two points in homogeneous coordinates'''
def cartesian_to_homogeneous(line):
    hom_line = np.cross([line[0], line[1], 1], [line[2], line[3], 1])
    hom_line = hom_line/hom_line[2]
    return hom_line

''' compute the intersection between lines and the reference line'''
def compute_intersections_between_lines(lines, reference_line):
    intersection_points_x = []
    # compute the intersection of each line with the horizontal line
    for line in lines:

        hom_line = cartesian_to_homogeneous(line[0])
        hom_ref_line = cartesian_to_homogeneous(reference_line)

        int_point = np.cross(hom_line, hom_ref_line)
        int_point = int_point / int_point[2]

        intersection_points_x.append(int_point[0])

    return intersection_points_x

''' Select the closest left and right line'''
def select_closest_lines(lines, horizontal_line, center, null_line=None):
    # compute the intersection of the lines with the horizontal lines
    intersections_points_x = compute_intersections_between_lines(lines, horizontal_line)
    left_lines = []
    right_lines = []
    left_distances = [] 
    right_distances = []
    for i in range(len(lines)):
        # if the intersection in at the left of the center
        if intersections_points_x[i] < center:
            left_lines.append(lines[i])
            left_distances.append(abs(center - intersections_points_x[i]))

        else: # if the intersection is at the right of the center
            right_lines.append(lines[i])
            right_distances.append(abs(center - intersections_points_x[i]))

    #compute the indeces of the minimum distance point
    min_left_index = left_distances.index(min(left_distances)) if left_distances else None
    min_right_index = right_distances.index(min(right_distances)) if right_distances else None

    # if exists, return the lines closest to the point
    if min_left_index is None:
        if min_right_index is None:
            return null_line, null_line
        return null_line, right_lines[min_right_index]
    
    if min_right_index is None:
        return left_lines[min_left_index], null_line

    return left_lines[min_left_index], right_lines[min_right_index]

''' Filter out 'quite horizontal' lines and lines below the horizotal,
    then select the closest left and right line'''
def filter_lines(lines_p, horizontal_line, image_center, frame_height, tolerance_angle = 20):
    # Calculate the homogeneous coordinates of the horizontal line
    x1, y1, x2, y2 = horizontal_line
    horizontal_line_homogeneous = np.cross([x1, y1, 1], [x2, y2, 1])        
    horizontal_line_homogeneous = horizontal_line_homogeneous / horizontal_line_homogeneous[0]

    # Filter out lines that are 'quite horizontal' with a tolerance of 20 degrees
    filtered_lines = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            angle = calculate_angle(x1, y1, x2, y2)
            if abs(angle) > tolerance_angle:
                y_max = max(y1, y2)
                y_min = min(y1, y2)
                x_max = x1 if y_max == y1 else x2
                # Filter the lines that have both endpoints over the horizontal line
                if x_max  + y_max * horizontal_line_homogeneous[1] + horizontal_line_homogeneous[2] > 0 and y_min > frame_height / 4: # se a*x + b*y + c > 0 allora il punto è sopra la linea
                    filtered_lines.append(line)

    # define the null line
    null_line = np.array([[0, 0, 0, 0]])

    if len(filtered_lines) == 0:
        return null_line, null_line

    # divide the lines in left and right and select the closests
    left_line, right_line = select_closest_lines(filtered_lines, horizontal_line, image_center, null_line)
 
    return left_line, right_line

''' Compute a left and a right lateral line for each frame'''
def compute_lateral_lines(frame, horizontal_line):
    
    # define the central point on the horizontal line
    central_point = select_central_point(horizontal_line, frame)

    # compute the edges
    edges = get_edges(frame, blur=True)

    # parameters for PHoughTransform
    min_line_length = 50
    max_line_gap = 5 
    # get the lateral lines
    lines_p = get_lines_pht(edges, min_line_length, max_line_gap)

    # select the closest left an right line
    left_line, right_line = filter_lines(lines_p, horizontal_line, central_point[0], frame.shape[0]) 

    return left_line, right_line

''' Get the lateral lines from the video'''
def get_lateral_lines(cap, bottom_lines):
    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Loop through each frame in the video
    frame_index = 0
    left_lines = []
    right_lines = []
    while frame_index < len(bottom_lines):
        ret, video_frame = cap.read()
        if not ret:
            print("Failed to read the frame at iteration (Lateral lines detection)", frame_index)
            break

        # Compute the three lines in the frame
        left_line, right_line = compute_lateral_lines(video_frame, bottom_lines[frame_index]) 

        # Append the lines to the lists
        left_lines.append(left_line[0])
        right_lines.append(right_line[0])

        # Increment the frame index
        frame_index += 1

    return left_lines, right_lines

# ==============================================================================
#                            POST PROCESSING LATERAL LINES FUNCTIONS
# ==============================================================================

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


# ==============================================================================
#                              UPPER LINE DETECTION FUNCTIONS
# ==============================================================================

''' Compute the determinant '''
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

''' Get the intersection point of two lines'''
def get_intersection(line_1, line_2):
    xdiff = (line_1[0] - line_1[2], line_2[0] - line_2[2])
    ydiff = (line_1[1] - line_1[3], line_2[1] - line_2[3])

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det((line_1[0], line_1[1]), (line_1[2], line_1[3])), 
           det((line_2[0], line_2[1]), (line_2[2], line_2[3])))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    return x, y

''' Calculate the distance between the two intersection points '''
def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def cut_frame_triangle(frame, bottom_line, left_line, rigth_line):
    """Cut the image based on the lines defined in the DataFrame."""
    width = frame.shape[1]
    height = frame.shape[0]
    # --- Get extended lines ---
    bottom_line = get_extended_line(bottom_line, width, height)
    left_line = get_extended_line(left_line, width, height)
    rigth_line = get_extended_line(rigth_line, width, height)

    # --- Get triangle intersection points ---
    int1 = get_intersection(bottom_line, left_line)
    int2 = get_intersection(bottom_line, rigth_line)
    int3 = get_intersection(left_line, rigth_line)

    if None in [int1, int2, int3]:
        raise ValueError("Could not find all three triangle points")

    triangle = np.array([int1, int2, int3])

    # --- Create mask and apply it ---
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [triangle], 0, 255, -1)

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame, triangle

''' Extraxt the brown and rose colors'''
def extract_br_frame(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Brown range
    lower_brown = np.array([00, 00, 50])
    upper_brown = np.array([20, 255, 255])

    # Rose (pinkish-red) 
    lower_rose = np.array([150, 30, 200])
    upper_rose = np.array([180, 200, 255])

    # Create masks
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_rose = cv2.inRange(hsv, lower_rose, upper_rose)

    # Combine both masks
    combined_mask = cv2.bitwise_or(mask_brown, mask_rose)

    # Apply the combined mask to the original frame
    brown_and_rose_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    return brown_and_rose_frame

''' Get a first estimate of the upper line 
    by finding the y-point in the triangle where the horizontal line becomes 985 balck'''
def get_upper_horizontal_line_first_estimate(frame, triangle):
    # Threshold for black (treat anything darker than this as black)
    black_thresh = 30

    # Convert masked image to grayscale for easier intensity check
    gray_masked = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Start from row y = 100
    start_y = triangle[0][1]
    width = gray_masked.shape[1]
    stop_row = None

    for y in range(start_y, -1, -1):  # go from int1 down to 0
        row = gray_masked[y, :]
        non_black_pixels = np.count_nonzero(row > black_thresh)
        percentage_non_black = (non_black_pixels / len(row)) * 100

        if percentage_non_black < 2:
            stop_row = y
            break

    if stop_row is None:
        print("No row found with <2% non-black pixels after y=100.")
        return None

    #  Define the horizontal line  
    horizontal_upper_line =[0, y, width, y]

    return horizontal_upper_line

''' Apply template matching too tdetect the bottom point of the pins'''
def template_matching(br_frame, template, upper_horizontal_estimated, left_line, right_line):
    # computed estimated intersection between upper and lateral lines
    intersection_left = get_intersection(left_line, upper_horizontal_estimated)
    intersection_right = get_intersection(right_line, upper_horizontal_estimated)
    # compute the length of the upper line in the frame
    distance = euclidean_distance(intersection_left, intersection_right)

    # Compute the correct dimension of the template
    lane_width = 1066
    pin_height_real = 381 + 40 # 20 is the margin taken from the template
    pin_height_template = template.shape[0]
    pin_width_template = template.shape[1]

    pin_height = (pin_height_real * distance) / lane_width
    f = pin_height / pin_height_template

    template = cv2.resize(template, (0, 0), fx=f, fy=f)

    new_width = int(pin_width_template * f)
    new_height = int(pin_height_template * f)


    # --- Template matching ---
    gray_frame = cv2.cvtColor(br_frame, cv2.COLOR_BGR2GRAY)

    # Method for doing Template Matching
    method = cv2.TM_CCOEFF

    img = gray_frame.copy()
    result = cv2.matchTemplate(img, template, method) # This performs Convolution, the output will be (Width - w + 1, Height - h + 1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # This returns min, max values, min, max locations
    location = max_loc
        
    bottom_right = (location[0] + new_width, location[1] + new_height)


    return bottom_right

"""Extend a line to the image boundaries."""
def get_extended_line(line, img_width=2000, img_height=2000):
    
    x1, y1, x2, y2 = line
    if x1 == x2:
        return (x1, 0), (x2, img_height)
    
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    points = []

    y_left = int(m * 0 + b)
    y_right = int(m * img_width + b)
    if 0 <= y_left <= img_height:
        points.append((0, y_left))
    if 0 <= y_right <= img_height:
        points.append((img_width, y_right))

    if m != 0:
        x_top = int((0 - b) / m)
        x_bottom = int((img_height - b) / m)
        if 0 <= x_top <= img_width:
            points.append((x_top, 0))
        if 0 <= x_bottom <= img_width:
            points.append((x_bottom, img_height))
    extended_line = [points[0][0], points[0][1], points[1][0], points[1][1]]

    return extended_line if len(points) >= 2 else [x1, y1, x2, y2]

''' correct the inclination of the founded upper line'''
def correct_inclination(bottom_right, bottom_line, frame):
    # set the line horizontal
    ux1 = bottom_right[0]
    uy1 = bottom_right[1]
    ux2 = bottom_right[0]+100
    uy2 = bottom_right[1]

    # Calculate the extended line points
    width = frame.shape[1]
    height = frame.shape[0]
    upper_line = get_extended_line([ux1, uy1, ux2, uy2], width, height)
    return upper_line

''' Compute the upper line in a single frame'''
def compute_upper_line(frame, template, bottom_line, left_line, right_line):
    cutted_frame, triangle = cut_frame_triangle(frame, bottom_line, left_line, right_line)
    br_frame = extract_br_frame(cutted_frame)
    upper_horizontal_estimated = get_upper_horizontal_line_first_estimate(br_frame, triangle)
    bottom_rigth_point_pin = template_matching(cutted_frame, template, upper_horizontal_estimated, left_line, right_line)
    upper_line = correct_inclination(bottom_rigth_point_pin, bottom_line, frame) 
    return upper_line

''' Compute the upper lines from the bottom and lateral lines'''
def get_upper_lines(cap, template_path, bottom_lines, left_lines, right_lines):
    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # get template
    template = cv2.imread(template_path, 0) # 0 for the grayscale image

    # Loop through each frame in the video
    frame_index = 0
    upper_lines = []
    while frame_index < len(bottom_lines):
        ret, video_frame = cap.read()
        if not ret:
            print("Failed to read the frame at iteration (Lateral linesdetection)", frame_index)
            break

        # Compute the three lines in the frame
        upper_line = compute_upper_line(frame=video_frame, template=template, bottom_line=bottom_lines[frame_index], left_line=left_lines[frame_index], right_line=right_lines[frame_index]) 

        # Append the lines to the lists
        upper_lines.append(upper_line)

        # Increment the frame index
        frame_index += 1
    return upper_lines

# ==============================================================================
#                          CREATE POINTS DF
# ==============================================================================
''' Create a df of the 4 corners of the lane from the 4 lines'''
def create_points_df(bottom_lines, left_lines, right_lines, top_lines):
    columns = ['Frame', 'bottom_left_x', 'bottom_left_y', 'bottom_right_x', 'bottom_right_y', 'up_left_x', 'up_left_y', 'up_right_x', 'up_right_y']
    data = []
    max_iterations = min(len(bottom_lines), len(left_lines), len(right_lines), len(top_lines))
    for i in range(max_iterations):        
        bottom_left = get_intersection(bottom_lines[i], left_lines[i])
        bottom_right = get_intersection(bottom_lines[i], right_lines[i])
        up_left = get_intersection(top_lines[i], left_lines[i])
        up_right = get_intersection(top_lines[i], right_lines[i])

        data.append([i, bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1], up_left[0], up_left[1], up_right[0], up_right[1]])

    # Create the DataFrame
    points_df = pd.DataFrame(data, columns=columns)

    return points_df


# ==============================================================================
#                  POST PROCESSING UPPER AND BOTTOM LINES FUNCTIONS
# ==============================================================================

'''if both are above return them, otherwise None '''
def is_disappeared(bl_0, br_0, bl, br, tr, tl, max_y, threshold=0.99):
    if bl[1] < max_y*threshold and br[1] < max_y*threshold: # both points are above the threshold -> keep them
        return bl, br
    return None, None

''' Remove outliers and compute the missing coordinates'''
def postprocessing_upper(points_df, bottom_y_distances):
    df = points_df[['Frame', 'up_left_y']].copy()
    df_length = len(df)

    distances = np.diff(df['up_left_y'].values)

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
        distances = np.diff(df['up_left_y'].values)

    # interpolate to found the missing values
    df['up_left_y'] = df['up_left_y'].interpolate(method='linear', limit_direction='both')

    # fill the remaining values at the end of the df with estimated values from the bottom line
    for i in range(len(df), df_length):
        if i >= len(df):
            df = pd.concat([df, pd.DataFrame({'Frame': [i], 'up_left_y': [df.loc[i-1, 'up_left_y'] + bottom_y_distances[i-1]]})], ignore_index=True)
    
    # update points_df with the new values
    for i, row in df.iterrows():
        y = row['up_left_y']
        left_intersection = get_intersection(
            [points_df.loc[i, 'bottom_left_x'], points_df.loc[i, 'bottom_left_y'],
            points_df.loc[i, 'up_left_x'], points_df.loc[i, 'up_left_y']],
            [0, y, 1000, y]
        )
        right_intersection = get_intersection(
            [points_df.loc[i, 'bottom_right_x'], points_df.loc[i, 'bottom_right_y'],
            points_df.loc[i, 'up_right_x'], points_df.loc[i, 'up_right_y']],
            [0, y, 1000, y]
        )
        if left_intersection is not None and right_intersection is not None:
            points_df.loc[i, 'up_left_x'] = left_intersection[0]
            points_df.loc[i, 'up_left_y'] = left_intersection[1]
            points_df.loc[i, 'up_right_x'] = right_intersection[0]
            points_df.loc[i, 'up_right_y'] = right_intersection[1]
    return points_df

''' When the bottom line is visible it adjust the top line
when it is not visible anymore, it compute the bottom line starting from the top one'''
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
        bl_prev = (df_copy.iloc[i-1]["bottom_left_x"], df_copy.iloc[i-1]["bottom_left_y"])
        br_prev = (df_copy.iloc[i-1]["bottom_right_x"], df_copy.iloc[i-1]["bottom_right_y"])
        tl_prev = (df_copy.iloc[i-1]["up_left_x"], df_copy.iloc[i-1]["up_left_y"])
        tr_prev = (df_copy.iloc[i-1]["up_right_x"], df_copy.iloc[i-1]["up_right_y"])
        bl = (points_df.iloc[i]["bottom_left_x"], points_df.iloc[i]["bottom_left_y"])
        br = (points_df.iloc[i]["bottom_right_x"], points_df.iloc[i]["bottom_right_y"])
        tr = (points_df.iloc[i]["up_right_x"], points_df.iloc[i]["up_right_y"])
        tl = (points_df.iloc[i]["up_left_x"], points_df.iloc[i]["up_left_y"])

        # case 1: the bottom line is visible
        if not bottom_disappeared:
            # select data in a sliding window
            window_size = 9
            if i < window_size:
                left_relative_position = (tl_prev[0] - bl_prev[0], tl_prev[1] - bl_prev[1])
                right_relative_position = (tr_prev[0] - br_prev[0], tr_prev[1] - br_prev[1])
            else:
                left_relative_positions = [(df_copy.iloc[j]["up_left_x"] - df_copy.iloc[j]["bottom_left_x"],
                                            df_copy.iloc[j]["up_left_y"] - df_copy.iloc[j]["bottom_left_y"]) 
                                           for j in range(i - window_size + 1, i + 1)]
                right_relative_positions = [(df_copy.iloc[j]["up_right_x"] - df_copy.iloc[j]["bottom_right_x"],
                                             df_copy.iloc[j]["up_right_y"] - df_copy.iloc[j]["bottom_right_y"]) 
                                            for j in range(i - window_size + 1, i + 1)]

                # select the second lower line in the frame
                left_relative_position = sorted(left_relative_positions, key=lambda pos: pos[1], reverse=True)[2]
                right_relative_position = sorted(right_relative_positions, key=lambda pos: pos[1], reverse=True)[2]
           
            # compute the new position of the bottom points in the current frame (if needed)
            bl_new, br_new = is_disappeared(bl_prev, br_prev, bl, br, tr, tl, height)

            if bl_new is None and br_new is None:
                bottom_disappeared = True
                index_disappeared = i
                print(f"Bottom points disappeared at frame {i}.")
            else: # consider correct the bottom points
                tr_mid = (br_new[0] + right_relative_position[0], br_new[1] + right_relative_position[1])
                tl_mid = (bl_new[0] + left_relative_position[0], bl_new[1] + left_relative_position[1])
                # Calculate the intersection point
                tr_new = get_intersection([tr_mid[0], tr_mid[1], tl_mid[0], tl_mid[1]], [br_new[0], br_new[1], tr[0], tr[1]])
                tl_new = get_intersection([tl_mid[0], tl_mid[1], tr_mid[0], tr_mid[1]], [bl_new[0], bl_new[1], tl[0], tl[1]])
                
        if bottom_disappeared: # consider correct the top poits
            bl_new = (tl[0] - left_relative_position[0], tl[1] - left_relative_position[1])
            br_new = (tr[0] - right_relative_position[0], tr[1] - right_relative_position[1])
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
            bl_new = (df.iloc[i]["up_left_x"] - left_relative_position[0], df.iloc[i]["up_left_y"] - left_relative_position[1])
            br_new = (df.iloc[i]["up_right_x"] - right_relative_position[0], df.iloc[i]["up_right_y"] - right_relative_position[1])

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
    mean_value = df_copy.iloc[:len(df_copy) // 2]["up_left_y"].mean()

    # Compute the intersection between the horizontal line with y coordinate equal to the mean value and the laetral lines
    for i in range(len(df_copy)):
        # Compute the intersection points
        left_intersection = get_intersection(
            [points_df.iloc[i]["bottom_left_x"], points_df.iloc[i]["bottom_left_y"],
            points_df.iloc[i]["up_left_x"], points_df.iloc[i]["up_left_y"]],
            [0, mean_value, 1000, mean_value]
        )
        right_intersection = get_intersection(
            [points_df.iloc[i]["bottom_right_x"], points_df.iloc[i]["bottom_right_y"],
            points_df.iloc[i]["up_right_x"], points_df.iloc[i]["up_right_y"]],
            [0, mean_value, 1000, mean_value]
        )
        if left_intersection is not None and right_intersection is not None:
            df_copy.at[i, "up_left_x"] = left_intersection[0]
            df_copy.at[i, "up_left_y"] = left_intersection[1]
            df_copy.at[i, "up_right_x"] = right_intersection[0]
            df_copy.at[i, "up_right_y"] = right_intersection[1]

    return df_copy

# ==============================================================================
#                              GENERATE THE VIDEO
# ==============================================================================

''' draw one line on the frame'''
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

''' draw the lines on the frame'''
def draw_lines_on_frame(frame, lines):
    for i in range(len(lines)):
        frame = draw_line_on_frame(frame, lines[i])
    return frame

''' Generate the video with the lines'''
def generate_video_lines(cap, output_path, points_df):
    # start the video from the begnning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # Loop through each frame in the video
    frame_index = 0
    while frame_index < len(points_df):
        ret, video_frame = cap.read()
        if not ret:
            print("Failed to read the frame at iteration (Generate video)", frame_index)
            break

        # draw the lines on the frame   
        lines = [
            [points_df.iloc[frame_index]["bottom_left_x"], points_df.iloc[frame_index]["bottom_left_y"], points_df.iloc[frame_index]["bottom_right_x"], points_df.iloc[frame_index]["bottom_right_y"]],
            [points_df.iloc[frame_index]["up_left_x"], points_df.iloc[frame_index]["up_left_y"], points_df.iloc[frame_index]["up_right_x"], points_df.iloc[frame_index]["up_right_y"]],
            [points_df.iloc[frame_index]["bottom_left_x"], points_df.iloc[frame_index]["bottom_left_y"], points_df.iloc[frame_index]["up_left_x"], points_df.iloc[frame_index]["up_left_y"]],
            [points_df.iloc[frame_index]["bottom_right_x"], points_df.iloc[frame_index]["bottom_right_y"], points_df.iloc[frame_index]["up_right_x"], points_df.iloc[frame_index]["up_right_y"]]
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

# ==============================================================================
#                              GENERATE THE CSV FILE
# ==============================================================================

def publish_csv_lane_points(output_path, output_df):
    # Save the DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)
    print(f"CSV file with the lane points saved to {output_path}")

    return

# ==============================================================================
#                              OVERVIEW FUNCTION
# ==============================================================================

'''Organize the code into 4 working area + generate video + publish the lines in a csv'''
def get_lane_points(video_path: str, output_path_video: str, output_path_data: str, template_path: str):
    cap = cv2.VideoCapture(video_path)
    avg_motion = estimate_background_motion(cap)
    bottom_lines_raw = get_bottom_lines(cap)
    bottom_lines = postprocessing_bottom_lines(bottom_lines_raw, avg_motion)
    left_lines_raw, right_lines_raw = get_lateral_lines(cap, bottom_lines)
    left_lines, right_lines = postprocessing_lateral_lines(left_lines_raw, right_lines_raw, avg_motion)
    upper_lines_raw = get_upper_lines(cap, template_path, bottom_lines, left_lines, right_lines)
    points_df = create_points_df(bottom_lines, left_lines, right_lines, upper_lines_raw)
    if avg_motion > 1:
        points_df = postprocessing_top_bottom(points_df, cap)
    else:
        points_df = postprocessing_top_still(points_df)
    generate_video_lines(cap, output_path_video, points_df)
    publish_csv_lane_points(output_path_data, points_df)
    return

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()

    video_number = "2"
    PROJECT_ROOT = Path().resolve()
    video_path = str(PROJECT_ROOT / "data" / f"recording_{video_number}" / f"Recording_{video_number}.mp4")
    template_path = str(PROJECT_ROOT / "data" / "auxiliary_data" / "pin_template" / "Template_pin_3.png")
    output_path_video = str(PROJECT_ROOT / "data" / f"recording_{video_number}" / "Lane_detection.mp4")
    output_path_data = str(PROJECT_ROOT / "data" / "auxiliary_data" / "lane_points" / f"Lane_points_{video_number}.csv")
    print('Start lines detection of video:', video_number)
    get_lane_points(video_path, output_path_video, output_path_data, template_path)
    print('End lines detection of video:', video_number)

    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"Total runtime of Lane Detection: {elapsed_time:.2f} seconds")