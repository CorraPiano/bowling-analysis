import streamlit as st
import tempfile
import shutil
import subprocess
import cv2
from pathlib import Path
import base64


# ─── Global variables ─────────────────────────────────────────────────────────────

# Output directory (will be created on startup)
out_dir = Path("app_output_data/recording")
PROJECT_ROOT = Path().resolve().parent

# Lane detection
TEMPLATE_PIN_PATH       = str(PROJECT_ROOT / "output_data" / "templates" / "Template_pin.png")
VIDEO_LANE_DETECTION_PATH = str(out_dir / "Lane_detection.mp4")
LANE_POINTS_PATH          = str(out_dir / "Lane_points.csv")

# Ball detection
VIDEO_BALL_DETECTION_PATH    = str(out_dir / "Ball_detected_raw.mp4")
BALL_COORD_PATH              = str(out_dir / "Circle_positions_raw.csv")
BALL_COORD_CLEAR_PATH        = str(out_dir / "Circle_positions_cleaned.csv")
BALL_COORD_TRANS_PATH        = str(out_dir / "Transformed_positions_raw.csv")
BALL_COORD_TRANS_CLEAR_PATH  = str(out_dir / "Transformed_positions_processed.csv")
VIDEO_TRAJ_ON_RECORDING      = str(out_dir / "Tracked_output.mp4")
BALL_LOWER_COORD_PATH        = str(out_dir / "Ball_lower_point_raw.csv")
BALL_LOWER_COORD_CLEAN_PATH  = str(out_dir / "Adjusted_positions.csv")
VIDEO_BALL_PROCESSED_PATH    = str(out_dir / "Ball_detected_processed.mp4")

# Reconstruction & trajectory
BALL_COORD_DEFORMED_PATH     = str(out_dir / "Transformed_positions_deformed.csv")
TEMPLATE_LANE_PATH           = str(PROJECT_ROOT / "output_data" / "templates" / "Template_lane.png")
VIDEO_TRAJ_ON_LANE           = str(out_dir / "Reconstructed_trajectory_processed.mp4")
VIDEO_TRAJ_ON_LANE_DEFORMED  = str(out_dir / "Reconstructed_trajectory_deformed.mp4")

# Spin
ROTATION_DATA_PATH           = str(out_dir / "Rotation_data.csv")
ROTATION_DATA_PROCESSED_PATH = str(out_dir / "Rotation_data_processed.csv")
VIDEO_SPHERE_RAW_PATH        = str(out_dir / "Rotating_sphere_raw.mp4")
VIDEO_SPHERE_PATH            = str(out_dir / "Rotating_sphere.mp4")

# Final video
VIDEO_FINAL_PATH             = str(out_dir / "Final.mp4")

# ─── Pipeline imports ──────────────────────────────────────────────────────────────

from lane_detection.Background_Motion import estimate_background_motion
from lane_detection.Bottom_Line_Detection import get_bottom_lines
from lane_detection.Bottom_Line_Postprocessing import postprocessing_bottom_lines
from lane_detection.Lateral_Lines_detection import get_lateral_lines
from lane_detection.Lateral_Lines_Postprocessing import postprocessing_lateral_lines
from lane_detection.Upper_Line_Detection import get_upper_lines
from lane_detection.Upper_Line_Postprocessing import (
    postprocessing_upper_lines, publish_csv_lane_points, generate_video_lines
)

from ball_detection.Detection import process_video_with_roi
from ball_detection.Post_processing_outliers import process_data
from ball_detection.Post_processing_smoothing import process_coordinates_final

from reconstruction.Post_processing_positions import process_data_transformed
from reconstruction.Reconstruction import process_reconstruction
from reconstruction.Reconstruction_deformed import process_reconstruction_deformed

from trajectory.Trajectory_on_video import trajectory_on_video
from trajectory.Trajectory_on_reconstruction import trajectory_on_reconstruction
from trajectory.Trajectory_on_reconstruction_deformed import trajectory_on_reconstruction_deformed

from spin.Detection import process_spin
from spin.Post_processing import spin_post_processing
from spin.Video_creation import spin_video_creation

from utility.Final_video_creation import create_final_video

# ─── Helpers ──────────────────────────────────────────────────────────────────────

def ensure_out_dir():
    """Create the output directory if it doesn’t exist."""
    out_dir.mkdir(parents=True, exist_ok=True)

def save_uploaded_video(uploaded) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(tmp.name, "wb") as f:
        shutil.copyfileobj(uploaded, f)
    return tmp.name

def reencode_for_web(input_mp4: str, output_mp4: str):
    """Wrap into baseline H.264 so browsers will play it."""
    cmd = [
        "ffmpeg", "-y", "-i", input_mp4,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
        "-movflags", "+faststart",
        output_mp4
    ]
    subprocess.run(cmd, check=True)


def display_video_with_max_height(path: str, max_height_px: int = 400):
    """Read a local MP4 and embed it as a <video> with CSS max-height."""
    video_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(video_bytes).decode()
    html = f"""
    <video controls
           style="max-height:{max_height_px}px; width:auto; display:block; margin:auto;">
      <source src="data:video/mp4;base64,{b64}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(html, unsafe_allow_html=True)

# ─── Stage functions ─────────────────────────────────────────────────────────────

def run_lane_detection(input_path):
    cap = cv2.VideoCapture(input_path)
    avg_motion = estimate_background_motion(cap)
    bottom_raw = get_bottom_lines(cap)
    bottom = postprocessing_bottom_lines(bottom_raw, avg_motion)
    left_raw, right_raw = get_lateral_lines(cap, bottom)
    left, right = postprocessing_lateral_lines(left_raw, right_raw, avg_motion)
    upper_raw = get_upper_lines(cap, TEMPLATE_PIN_PATH, bottom, left, right)
    pts = postprocessing_upper_lines(cap, bottom, left, right, upper_raw, avg_motion)

    generate_video_lines(cap, VIDEO_LANE_DETECTION_PATH, pts)
    publish_csv_lane_points(LANE_POINTS_PATH, pts)

    # produce web-ready file
    lane_web = str(out_dir / "lane_web.mp4")
    reencode_for_web(VIDEO_LANE_DETECTION_PATH, lane_web)
    return lane_web

def run_ball_detection(input_path, lane_csv):
    process_video_with_roi(input_path, lane_csv,
                           VIDEO_BALL_DETECTION_PATH, BALL_COORD_PATH)
    process_data(BALL_COORD_PATH, BALL_COORD_CLEAR_PATH)
    process_reconstruction(lane_csv, BALL_COORD_CLEAR_PATH, BALL_COORD_TRANS_PATH)
    process_data_transformed(BALL_COORD_TRANS_PATH, BALL_COORD_TRANS_CLEAR_PATH)
    trajectory_on_video(input_path,
                        BALL_COORD_TRANS_CLEAR_PATH, lane_csv,
                        VIDEO_TRAJ_ON_RECORDING, BALL_LOWER_COORD_PATH)
    process_coordinates_final(input_path,
                              BALL_COORD_CLEAR_PATH, BALL_LOWER_COORD_PATH,
                              BALL_LOWER_COORD_CLEAN_PATH, VIDEO_BALL_PROCESSED_PATH)

    ball_web = str(out_dir / "ball_web.mp4")
    reencode_for_web(VIDEO_BALL_PROCESSED_PATH, ball_web)
    return ball_web, BALL_COORD_CLEAR_PATH

def run_reconstruction_and_trajectory(input_path, ball_csv_clean):
    # reconstruction steps
    process_reconstruction(LANE_POINTS_PATH, BALL_COORD_CLEAR_PATH, BALL_COORD_TRANS_PATH)
    process_data_transformed(BALL_COORD_TRANS_PATH, ball_csv_clean)
    process_reconstruction_deformed(ball_csv_clean,
                                   BALL_COORD_DEFORMED_PATH,
                                   TEMPLATE_LANE_PATH)

    # trajectory steps
    trajectory_on_video(input_path,
                        BALL_COORD_TRANS_CLEAR_PATH, LANE_POINTS_PATH,
                        VIDEO_TRAJ_ON_RECORDING, BALL_LOWER_COORD_PATH)
    trajectory_on_reconstruction(input_path,
                                 BALL_COORD_TRANS_CLEAR_PATH,
                                 VIDEO_TRAJ_ON_LANE)
    trajectory_on_reconstruction_deformed(input_path,
                                          BALL_COORD_DEFORMED_PATH,
                                          TEMPLATE_LANE_PATH,
                                          VIDEO_TRAJ_ON_LANE_DEFORMED)

    traj_web = str(out_dir / "trajectory_web.mp4")
    reencode_for_web(VIDEO_TRAJ_ON_LANE_DEFORMED, traj_web)
    return traj_web, VIDEO_TRAJ_ON_RECORDING, BALL_COORD_DEFORMED_PATH

def run_spin_analysis(input_path):
    process_spin(input_path, BALL_LOWER_COORD_CLEAN_PATH, ROTATION_DATA_PATH)
    spin_post_processing(ROTATION_DATA_PATH,
                         ROTATION_DATA_PROCESSED_PATH,
                         BALL_LOWER_COORD_CLEAN_PATH,
                         input_path)
    spin_video_creation(input_path,
                        VIDEO_SPHERE_RAW_PATH,
                        VIDEO_SPHERE_PATH,
                        ROTATION_DATA_PROCESSED_PATH)

    spin_web = str(out_dir / "spin_web.mp4")
    reencode_for_web(VIDEO_SPHERE_PATH, spin_web)
    return spin_web

def run_final_video(lane_vid, traj_vid, spin_vid):
    create_final_video(lane_vid, traj_vid, spin_vid, VIDEO_FINAL_PATH)
    final_web = str(out_dir / "final_web.mp4")
    reencode_for_web(VIDEO_FINAL_PATH, final_web)
    return final_web

# ─── Streamlit app ───────────────────────────────────────────────────────────────

def main():
    ensure_out_dir()

    st.set_page_config(page_title="Bowling Analysis Pipeline",
                        page_icon=None, layout="centered",
                        initial_sidebar_state="expanded",
                        menu_items={
                            'Get Help': 'https://github.com/CorraPiano/bowling-analysis',
                            'Report a bug': None,
                            'About': "Bowling Analysis Pipeline"
                        })
    st.title("🎥 Bowling Analysis Pipeline")

    # Initialize session state keys
    for key in (
        "input_path",
        "lane_vid_web", "lane_csv",
        "ball_vid_web", "ball_csv_clean",
        "trajectory_vid_web", "traj_on_video", "deformed_csv",
        "spin_vid_web",
        "final_vid_web"
    ):
        st.session_state.setdefault(key, None)

    # 1️⃣ Upload
    uploaded = st.file_uploader("1️⃣ Upload recording", type=["mp4", "avi"])
    if uploaded:
        st.session_state.input_path = save_uploaded_video(uploaded)

    # 2️⃣ Lane Detection
    with st.expander("2️⃣ Lane Detection", expanded=True):
        if not st.session_state.input_path:
            st.info("Please upload a video to begin.")
        elif not st.session_state.lane_vid_web:
            if st.button("Run Lane Detection"):
                lane_web = run_lane_detection(st.session_state.input_path)
                st.session_state.lane_vid_web = lane_web
                st.session_state.lane_csv    = LANE_POINTS_PATH
                st.success("✅ Lane detection complete.")
        else:
            st.video(st.session_state.lane_vid_web)
            #display_video_with_max_height(st.session_state.lane_vid_web, max_height_px=400)

    # 3️⃣ Ball Detection
    with st.expander("3️⃣ Ball Detection"):
        if not st.session_state.lane_csv:
            st.info("Run lane detection first.")
        elif not st.session_state.ball_vid_web:
            if st.button("Run Ball Detection"):
                ball_web, clean_csv = run_ball_detection(
                    st.session_state.input_path,
                    st.session_state.lane_csv
                )
                st.session_state.ball_vid_web   = ball_web
                st.session_state.ball_csv_clean = clean_csv
                st.success("✅ Ball detection complete.")
        else:
            st.video(st.session_state.ball_vid_web)
            #display_video_with_max_height(st.session_state.ball_vid_web, max_height_px=400)


    # 4️⃣ Reconstruction & Trajectory
    with st.expander("4️⃣ Reconstruction & Trajectory"):
        if not st.session_state.ball_csv_clean:
            st.info("Run ball detection first.")
        elif not st.session_state.trajectory_vid_web:
            if st.button("Run Reconstruction + Trajectory"):
                traj_web, traj_on_video, deformed_csv = run_reconstruction_and_trajectory(
                    st.session_state.input_path,
                    st.session_state.ball_csv_clean
                )
                st.session_state.trajectory_vid_web = traj_web
                st.session_state.deformed_csv       = deformed_csv
                st.session_state.traj_on_video     = traj_on_video
                st.success("✅ Reconstruction & trajectory complete.")
        else:
            #st.video(st.session_state.trajectory_vid_web)
            display_video_with_max_height(st.session_state.trajectory_vid_web, max_height_px=500)

    # 5️⃣ Spin Analysis
    with st.expander("5️⃣ Spin Analysis"):
        if not st.session_state.trajectory_vid_web:
            st.info("Run reconstruction & trajectory first.")
        elif not st.session_state.spin_vid_web:
            if st.button("Run Spin Analysis"):
                spin_web = run_spin_analysis(st.session_state.input_path)
                st.session_state.spin_vid_web = spin_web
                st.success("✅ Spin analysis complete.")
        else:
            #st.video(st.session_state.spin_vid_web)
            display_video_with_max_height(st.session_state.spin_vid_web, max_height_px=400)

    # 6️⃣ Final Video
    with st.expander("6️⃣ Final Video"):
        if not st.session_state.spin_vid_web:
            st.info("Run spin analysis first.")
        elif not st.session_state.final_vid_web:
            if st.button("Compose Final Video"):
                final_web = run_final_video(
                    st.session_state.traj_on_video,
                    st.session_state.trajectory_vid_web,
                    st.session_state.spin_vid_web
                )
                st.session_state.final_vid_web = final_web
                st.success("✅ Final video ready.")
        else:
            st.video(st.session_state.final_vid_web)
            #display_video_with_max_height(st.session_state.final_vid_web, max_height_px=400)

if __name__ == "__main__":
    main()
