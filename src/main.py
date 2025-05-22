from pathlib import Path
import cv2

from lane_detection.Background_Motion import estimate_background_motion
from lane_detection.Bottom_Line_Detection import get_bottom_lines
from lane_detection.Bottom_Line_Postprocessing import postprocessing_bottom_lines
from lane_detection.Lateral_Lines_Postprocessing import postprocessing_lateral_lines
from lane_detection.Lateral_Lines_detection import get_lateral_lines
from lane_detection.Upper_Line_Detection import get_upper_lines
from lane_detection.Upper_Line_Postprocessing import postprocessing_upper_lines, publish_csv_lane_points, generate_video_lines 

from ball_detection.Detection import process_video_with_roi
from ball_detection.Post_processing_outliers import process_data
from reconstruction.Post_processing_positions import process_data_transformed
from reconstruction.Reconstruction import process_reconstruction
from reconstruction.Reconstruction_deformed import process_reconstruction_deformed
from spin.Detection import process_spin
from spin.Post_processing import spin_post_processing
from spin.Video_creation import spin_video_creation
from trajectory.Trajectory_on_reconstruction import trajectory_on_reconstruction
from trajectory.Trajectory_on_reconstruction_deformed import trajectory_on_reconstruction_deformed
from trajectory.Trajectory_on_video import trajectory_on_video
from ball_detection.Post_processing_smoothing import process_coordinates_final
from utility.Final_video_creation import create_final_video


#===================================================================================
# This script runs the entire pipeline for the ball analysis
#===================================================================================

if __name__ == "__main__":

    #===============================================================================
    # PATHS
    #===============================================================================
    VIDEO_NUM = "4"
    PROJECT_ROOT = Path().resolve()

    #Lane detection
    INPUT_VIDEO_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / f"Recording_{VIDEO_NUM}.mp4") 
    TEMPLATE_PIN_PATH = str(PROJECT_ROOT / "output_data" / "templates" / "Template_pin.png")
    VIDEO_LANE_DETECTION_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_video" / f"Lane_detection_{VIDEO_NUM}.mp4")
    LANE_POINTS_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Lane_points_{VIDEO_NUM}.csv")

    #Ball detection
    VIDEO_BALL_DETECTION_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_video" / f"Ball_detected_raw_{VIDEO_NUM}.mp4")
    BALL_COORD_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Circle_positions_raw_{VIDEO_NUM}.csv")
    BALL_COORD_CLEAR_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Circle_positions_cleaned_{VIDEO_NUM}.csv")
    BALL_COORD_TRANS_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Transformed_positions_raw_{VIDEO_NUM}.csv")
    BALL_COORD_TRANS_CLEAR_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Transformed_positions_processed_{VIDEO_NUM}.csv")
    VIDEO_TRAJ_ON_RECORDING = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_video" / f"Tracked_output_{VIDEO_NUM}.mp4")
    BALL_LOWER_COORD_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" /  f"Ball_lower_point_raw_{VIDEO_NUM}.csv")
    BALL_LOWER_COORD_CLEAN_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" /  f"Adjusted_positions_{VIDEO_NUM}.csv")
    VIDEO_BALL_PROCESSED_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_video" /  f"Ball_detected_processed_{VIDEO_NUM}.mp4")

    #Reconstruction
    BALL_COORD_DEFORMED_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Transformed_positions_deformed_{VIDEO_NUM}.csv")

    #Trajectory
    VIDEO_TRAJ_ON_LANE = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}"  / f"Reconstructed_trajectory_processed_{VIDEO_NUM}.mp4")
    TEMPLATE_LANE_PATH = str(PROJECT_ROOT / "output_data" / "templates" / "Template_lane.png")
    VIDEO_TRAJ_ON_LANE_DEFORMED = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / f"Reconstructed_trajectory_deformed_{VIDEO_NUM}.mp4")

    #Spin
    ROTATION_DATA_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Rotation_data_{VIDEO_NUM}.csv")
    ROTATION_DATA_PROCESSED_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_data" / f"Rotation_data_processed_{VIDEO_NUM}.csv")
    VIDEO_SPHERE_RAW_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / "other_video" / f"Rotating_sphere_raw_{VIDEO_NUM}.mp4")
    VIDEO_SPHERE_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}"  / "Rotating_sphere.mp4")

    #Video creation
    VIDEO_FINAL_PATH = str(PROJECT_ROOT / "output_data" / f"recording_{VIDEO_NUM}" / f"Final_{VIDEO_NUM}.mp4")



    #===============================================================================
    # LANE DETECTION
    #===============================================================================
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    avg_motion = estimate_background_motion(cap)
    bottom_lines_raw = get_bottom_lines(cap)
    bottom_lines = postprocessing_bottom_lines(bottom_lines_raw, avg_motion)
    left_lines_raw, right_lines_raw = get_lateral_lines(cap, bottom_lines)
    left_lines, right_lines = postprocessing_lateral_lines(left_lines_raw, right_lines_raw, avg_motion)
    upper_lines_raw = get_upper_lines(cap, TEMPLATE_PIN_PATH, bottom_lines, left_lines, right_lines)
    points_df = postprocessing_upper_lines(cap, bottom_lines, left_lines, right_lines, upper_lines_raw, avg_motion)
    generate_video_lines(cap, VIDEO_LANE_DETECTION_PATH, points_df)
    publish_csv_lane_points(LANE_POINTS_PATH, points_df)

    #===============================================================================
    # BALL DETECTION
    #===============================================================================
    process_video_with_roi(INPUT_VIDEO_PATH, LANE_POINTS_PATH, VIDEO_BALL_DETECTION_PATH, BALL_COORD_PATH)
    process_data(BALL_COORD_PATH, BALL_COORD_CLEAR_PATH)
    process_reconstruction(LANE_POINTS_PATH, BALL_COORD_CLEAR_PATH, BALL_COORD_TRANS_PATH)
    process_data_transformed(BALL_COORD_TRANS_PATH, BALL_COORD_TRANS_CLEAR_PATH)
    trajectory_on_video(INPUT_VIDEO_PATH, BALL_COORD_TRANS_CLEAR_PATH, LANE_POINTS_PATH, VIDEO_TRAJ_ON_RECORDING, BALL_LOWER_COORD_PATH)
    process_coordinates_final(INPUT_VIDEO_PATH, BALL_COORD_CLEAR_PATH, BALL_LOWER_COORD_PATH, BALL_LOWER_COORD_CLEAN_PATH, VIDEO_BALL_PROCESSED_PATH)
    
    #===============================================================================
    # RECONSTRUCTION
    #===============================================================================
    #process_reconstruction(LANE_POINTS_PATH, BALL_COORD_CLEAR_PATH, BALL_COORD_TRANS_PATH)
    #process_data_transformed(BALL_COORD_TRANS_PATH, BALL_COORD_TRANS_CLEAR_PATH)
    process_reconstruction_deformed(BALL_COORD_TRANS_CLEAR_PATH, BALL_COORD_DEFORMED_PATH, TEMPLATE_LANE_PATH)

    #===============================================================================
    # TRAJECTORY
    #===============================================================================
    #trajectory_on_video(INPUT_VIDEO_PATH, BALL_COORD_TRANS_CLEAR_PATH, LANE_POINTS_PATH, VIDEO_TRAJ_ON_RECORDING, BALL_LOWER_COORD_PATH)
    trajectory_on_reconstruction(INPUT_VIDEO_PATH, BALL_COORD_TRANS_CLEAR_PATH, VIDEO_TRAJ_ON_LANE)
    trajectory_on_reconstruction_deformed(INPUT_VIDEO_PATH, BALL_COORD_DEFORMED_PATH, TEMPLATE_LANE_PATH, VIDEO_TRAJ_ON_LANE_DEFORMED)

    #===============================================================================
    # SPIN
    #===============================================================================
    process_spin(INPUT_VIDEO_PATH, BALL_LOWER_COORD_CLEAN_PATH, ROTATION_DATA_PATH)
    spin_post_processing(ROTATION_DATA_PATH, ROTATION_DATA_PROCESSED_PATH, BALL_LOWER_COORD_CLEAN_PATH, INPUT_VIDEO_PATH)
    spin_video_creation(INPUT_VIDEO_PATH, VIDEO_SPHERE_RAW_PATH, VIDEO_SPHERE_PATH, ROTATION_DATA_PROCESSED_PATH)

    #===============================================================================
    # FINAL VIDEO
    #===============================================================================
    create_final_video(VIDEO_TRAJ_ON_RECORDING, VIDEO_TRAJ_ON_LANE_DEFORMED, VIDEO_SPHERE_PATH, VIDEO_FINAL_PATH)