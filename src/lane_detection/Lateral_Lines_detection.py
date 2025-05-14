import cv2
import numpy as np

from lane_detection.Bottom_Line_Detection import (
    calculate_angle,
    get_edges,
    get_lines_pht,
)

""" select the point at the center (x) of the frame on the line"""


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


""" transform the line composed by two points in homogeneous coordinates"""


def cartesian_to_homogeneous(line):
    hom_line = np.cross([line[0], line[1], 1], [line[2], line[3], 1])
    hom_line = hom_line / hom_line[2]
    return hom_line


""" compute the intersection between lines and the reference line"""


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


""" Select the closest left and right line"""


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

        else:  # if the intersection is at the right of the center
            right_lines.append(lines[i])
            right_distances.append(abs(center - intersections_points_x[i]))

    # compute the indeces of the minimum distance point
    min_left_index = (
        left_distances.index(min(left_distances)) if left_distances else None
    )
    min_right_index = (
        right_distances.index(min(right_distances)) if right_distances else None
    )

    # if exists, return the lines closest to the point
    if min_left_index is None:
        if min_right_index is None:
            return null_line, null_line
        return null_line, right_lines[min_right_index]

    if min_right_index is None:
        return left_lines[min_left_index], null_line

    return left_lines[min_left_index], right_lines[min_right_index]


""" Filter out 'quite horizontal' lines and lines below the horizotal,
    then select the closest left and right line"""


def filter_lines(
    lines_p, horizontal_line, image_center, frame_height, tolerance_angle=20
):
    # Calculate the homogeneous coordinates of the horizontal line
    x1, y1, x2, y2 = horizontal_line
    horizontal_line_homogeneous = np.cross([x1, y1, 1], [x2, y2, 1])
    horizontal_line_homogeneous = (
        horizontal_line_homogeneous / horizontal_line_homogeneous[0]
    )

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
                if (
                    x_max
                    + y_max * horizontal_line_homogeneous[1]
                    + horizontal_line_homogeneous[2]
                    > 0
                    and y_min > frame_height / 4
                ):  # se a*x + b*y + c > 0 allora il punto è sopra la linea
                    filtered_lines.append(line)

    # define the null line
    null_line = np.array([[0, 0, 0, 0]])

    if len(filtered_lines) == 0:
        return null_line, null_line

    # divide the lines in left and right and select the closests
    left_line, right_line = select_closest_lines(
        filtered_lines, horizontal_line, image_center, null_line
    )

    return left_line, right_line


""" Compute a left and a right lateral line for each frame"""


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
    left_line, right_line = filter_lines(
        lines_p, horizontal_line, central_point[0], frame.shape[0]
    )

    return left_line, right_line


""" Get the lateral lines from the video"""


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
            print(
                "Failed to read the frame at iteration (Lateral lines detection)",
                frame_index,
            )
            break

        # Compute the three lines in the frame
        left_line, right_line = compute_lateral_lines(
            video_frame, bottom_lines[frame_index]
        )

        # Append the lines to the lists
        left_lines.append(left_line[0])
        right_lines.append(right_line[0])

        # Increment the frame index
        frame_index += 1

    return left_lines, right_lines
