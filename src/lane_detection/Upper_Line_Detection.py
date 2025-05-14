import cv2
import numpy as np

""" Compute the determinant """


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


""" Get the intersection point of two lines"""


def get_intersection(line_1, line_2):
    xdiff = (line_1[0] - line_1[2], line_2[0] - line_2[2])
    ydiff = (line_1[1] - line_1[3], line_2[1] - line_2[3])

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (
        det((line_1[0], line_1[1]), (line_1[2], line_1[3])),
        det((line_2[0], line_2[1]), (line_2[2], line_2[3])),
    )
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    return x, y


""" Calculate the distance between the two intersection points """


def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


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


""" Extraxt the brown and rose colors"""


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


""" Get a first estimate of the upper line 
    by finding the y-point in the triangle where the horizontal line becomes 985 balck"""


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
    horizontal_upper_line = [0, y, width, y]

    return horizontal_upper_line


""" Apply template matching too tdetect the bottom point of the pins"""


def template_matching(
    br_frame, template, upper_horizontal_estimated, left_line, right_line
):
    # computed estimated intersection between upper and lateral lines
    intersection_left = get_intersection(left_line, upper_horizontal_estimated)
    intersection_right = get_intersection(right_line, upper_horizontal_estimated)
    # compute the length of the upper line in the frame
    distance = euclidean_distance(intersection_left, intersection_right)

    # Compute the correct dimension of the template
    lane_width = 1066
    pin_height_real = 381 + 40  # 20 is the margin taken from the template
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
    result = cv2.matchTemplate(
        img, template, method
    )  # This performs Convolution, the output will be (Width - w + 1, Height - h + 1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
        result
    )  # This returns min, max values, min, max locations
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


""" correct the inclination of the founded upper line"""


def correct_inclination(bottom_right, bottom_line, frame):
    # set the line horizontal
    ux1 = bottom_right[0]
    uy1 = bottom_right[1]
    ux2 = bottom_right[0] + 100
    uy2 = bottom_right[1]

    # Calculate the extended line points
    width = frame.shape[1]
    height = frame.shape[0]
    upper_line = get_extended_line([ux1, uy1, ux2, uy2], width, height)
    return upper_line


""" Compute the upper line in a single frame"""


def compute_upper_line(frame, template, bottom_line, left_line, right_line):
    cutted_frame, triangle = cut_frame_triangle(
        frame, bottom_line, left_line, right_line
    )
    br_frame = extract_br_frame(cutted_frame)
    upper_horizontal_estimated = get_upper_horizontal_line_first_estimate(
        br_frame, triangle
    )
    bottom_rigth_point_pin = template_matching(
        cutted_frame, template, upper_horizontal_estimated, left_line, right_line
    )
    upper_line = correct_inclination(bottom_rigth_point_pin, bottom_line, frame)
    return upper_line


""" Compute the upper lines from the bottom and lateral lines"""


def get_upper_lines(cap, template_path, bottom_lines, left_lines, right_lines):
    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # get template
    template = cv2.imread(template_path, 0)  # 0 for the grayscale image

    # Loop through each frame in the video
    frame_index = 0
    upper_lines = []
    while frame_index < len(bottom_lines):
        ret, video_frame = cap.read()
        if not ret:
            print(
                "Failed to read the frame at iteration (Lateral linesdetection)",
                frame_index,
            )
            break

        # Compute the three lines in the frame
        upper_line = compute_upper_line(
            frame=video_frame,
            template=template,
            bottom_line=bottom_lines[frame_index],
            left_line=left_lines[frame_index],
            right_line=right_lines[frame_index],
        )

        # Append the lines to the lists
        upper_lines.append(upper_line)

        # Increment the frame index
        frame_index += 1
    return upper_lines
