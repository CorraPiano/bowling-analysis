from pathlib import Path
from notebook.ball_detection.Detection import process_video_with_roi


def test(input, output):
    print("Input:", input) 
    print("Output:", output)

if __name__ == "__main__":

    VIDEO_NUM = "2"
    PROJECT_ROOT = Path().resolve().parent
    INPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUM}" / f"Recording_{VIDEO_NUM}.mp4")
    INPUT_CSV_PATH = str(PROJECT_ROOT / "data" / "auxiliary_data" / "lane_points" / f"Lane_points_{VIDEO_NUM}.csv")
    OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "data" / f"recording_{VIDEO_NUM}" / f"Ball_detected_raw_TEST_{VIDEO_NUM}.mp4")
    OUTPUT_CSV_PATH = str(PROJECT_ROOT / "notebook" / "ball_detection" / "intermediate_data" / f"Circle_positions_raw_TEST_{VIDEO_NUM}.csv")


    process_video_with_roi(INPUT_VIDEO_PATH, INPUT_CSV_PATH, OUTPUT_VIDEO_PATH, OUTPUT_CSV_PATH)

    # df = process_bottom()
    # df = process_laterali(df)
    # df = process_up(df)
    # df = post_processing_final(df)

    # if video:
    #     process_video_lane(df)

    detection_ball(csv)
    ...
    if video:
        process_video_ball(df)