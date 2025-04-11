import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import medfilt

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

    outlier_threshold =  threshold * std_dist

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
    df_full = Savitzky_Golay_filter(df_full, polyorder=polyorder)
    
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
def postprocessing_bottom_lines(horizontal_lines: list) -> list:
    # get the intersection points
    intersection_points = get_relevant_points(horizontal_lines)

    # Create a DataFrame from intersection_points
    points_df = pd.DataFrame(intersection_points, columns=['X', 'Y'])
    points_df['Frame'] = range(len(points_df))
    # Reorder columns to have 'Frame', 'X', 'Y' 
    points_df = points_df[['Frame', 'X', 'Y']]


    # Remove outliers
    points_cleaned = remove_outliers_bottom(points_df)

    # Smooth the trajectory to reduce noise
    points_smoothed = median_filter(points_cleaned)

    # Interpolate missing coordinates
    points_interpolated = interpolate_missing_coordinates(points_smoothed, end_frame=len(points_df)-1, polyorder=1)

    # recover the lines from the points
    bottom_lines = points_to_lines(points_interpolated)

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
def filter_lines(lines_p, horizontal_line, image_center, tolerance_angle = 20):
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
                x_max = x1 if y_max == y1 else x2
                # Filter the lines that have both endpoints over the horizontal line
                if x_max  + y_max * horizontal_line_homogeneous[1] + horizontal_line_homogeneous[2] > 0: # se a*x + b*y + c > 0 allora il punto è sopra la linea
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
    left_line, right_line = filter_lines(lines_p, horizontal_line, central_point[0]) 

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
            print("Failed to read the frame at iteration (Lateral linesdetection)", frame_index)
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
''' Plot the trajectory of the points''' # DA CANCELLARE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def plot_trajectory(df, title):
    plt.figure(figsize=(8,6))
    plt.plot(df['X'], -df['Y'], 'o-', label='Cleaned Trajectory')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    plt.show()

''' execute an iterated algorithm for removing outliers,
    the exit threshold is set as two times the mean of the distances between points at the first iteration'''
def remove_outliers_iterated(df):
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
                exit_threshold = std
            else:
                exit_threshold = 2 * mean_dist

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

''' Postprocessing of one of the 2 lateral lines'''
def postprocessing_lateral_line(lines):
    # fin the point of the lines closet to the origin and create the proper dataframe
    closest_points = get_relevant_points(lines)
   
    # create a proper DataFrame
    closest_points_df = pd.DataFrame(closest_points, columns=['X', 'Y'])
    closest_points_df['Frame'] = range(len(closest_points_df))
    closest_points_df = closest_points_df[['Frame', 'X', 'Y']]

    # Remove outliers iterated
    points_cleaned = remove_outliers_iterated(closest_points_df)

    # Smooth the trajectory to reduce noise
    points_smoothed = median_filter(points_cleaned)

    # Interpolate missing coordinates
    points_interpolated = interpolate_missing_coordinates(points_smoothed, end_frame=len(lines)-1, polyorder=3)

    # recover the lines from the points
    processed_line = points_to_lines(points_interpolated)

    return processed_line 

''' Postprocessing of the lateral lines, done separately'''
def postprocessing_lateral_lines(left_lines, right_lines):
    left_lines_processed = postprocessing_lateral_line(left_lines)
    right_lines_processed = postprocessing_lateral_line(right_lines)
    return left_lines_processed, right_lines_processed


# ==============================================================================
#                              LATERAL LINE DETECTION FUNCTIONS
# ==============================================================================

def get_upper_lines(cap, template, bottom_lines, left_lines, right_lines):
    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Loop through each frame in the video
    frame_index = 0
    left_lines = []
    right_lines = []
    while frame_index < len(bottom_lines):
        ret, video_frame = cap.read()
        if not ret:
            print("Failed to read the frame at iteration (Lateral linesdetection)", frame_index)
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
#                              GENERATE THE VIDEO
# ==============================================================================

''' draw one line on the frame'''
def draw_line_on_frame(frame, line):
    # Create a copy of the original frame to draw the first line
    modified_frame = np.copy(frame)

    # Extract the first line's rho and theta
    if line is not None:
        x1, y1, x2, y2 = line
        
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
def generate_video(cap, output_path, bottom_lines, left_lines, right_lines):
    # start the video from the begnning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # Loop through each frame in the video
    frame_index = 0
    while frame_index < len(bottom_lines):
        ret, video_frame = cap.read()
        if not ret:
            print("Failed to read the frame at iteration (Generate video)", frame_index)
            break

        # draw the lines on the frame   
        modified_frame = draw_lines_on_frame(video_frame, [bottom_lines[frame_index], left_lines[frame_index], right_lines[frame_index]])

        # Write the modified frame to the output video
        out.write(modified_frame)

        # Increment the frame index
        frame_index += 1

    # Release the video capture and writer objects
    out.release()

    print(f"Video with three lines saved to {output_path}")

# ==============================================================================
#                              GENERATE THE CSV FILE
# ==============================================================================

def publish_csv_three_lines(output_path, bottom_lines, left_lines, right_lines):
    # initializing df
    output_df = pd.DataFrame(columns=['frame_number', 'horizontal', 'left', 'right'])
    # Loop through each frame in the video
    for i in range(len(bottom_lines)):
        # Append the lines to the DataFrame
        output_df = output_df.append({
            'frame_number': i,
            'horizontal': bottom_lines[i],
            'left': left_lines[i],
            'right': right_lines[i]
        }, ignore_index=True)

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)
    print(f"CSV file with three lines saved to {output_path}")

    return

# ==============================================================================
#                              OVERVIEW FUNCTION
# ==============================================================================

'''Organize the code into 4 working area + generate video + publish the lines in a csv'''
def get_three_lines(video_path: str, output_path_video: str, output_path_data: str, template_path: str):
    cap = cv2.VideoCapture(video_path)
    bottom_lines_raw = get_bottom_lines(cap)
    bottom_lines = postprocessing_bottom_lines(bottom_lines_raw)
    left_lines_raw, right_lines_raw = get_lateral_lines(cap, bottom_lines)
    left_lines, right_lines = postprocessing_lateral_lines(left_lines_raw, right_lines_raw)
    upper_lines_raw = get_upper_lines(cap, template_path, bottom_lines, left_lines, right_lines)
    upper_lines = postprocessing_upper_lines(upper_lines_raw)
    generate_video(cap, output_path_video, bottom_lines, left_lines, right_lines, upper)
    # publish_csv_three_lines(output_path_data, bottom_lines, left_lines, right_lines)
    cap.release()
    return

# ==============================================================================
#                                   MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":

    video_number = "7"
    PROJECT_ROOT = Path().resolve()
    video_path = str(PROJECT_ROOT / "data" / f"recording_{video_number}" / f"Recording_{video_number}.mp4")
    template_path = str(PROJECT_ROOT / "data" / "auxiliary_data" / "pin_templates" / "Template_pin_3.png")
    output_path_video = str(PROJECT_ROOT / "data" / f"recording_{video_number}" / "3_lines_video.mp4")
    output_path_data = str(PROJECT_ROOT / "data" / "auxiliary_data" / "processed_3_lines.mp4")
    print('Start lines detection of video:', video_number)
    get_three_lines(video_path, output_path_video, output_path_data)
    print('End lines detection of video:', video_number)