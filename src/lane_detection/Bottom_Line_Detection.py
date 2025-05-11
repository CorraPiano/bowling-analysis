import cv2
import numpy as np
import math


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

