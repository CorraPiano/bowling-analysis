import streamlit as st
import tempfile
import os
from main import test
import subprocess

# =================================================================
#                           STREAMLIT DEMO
# =================================================================

@st.cache_resource # Cache the function to avoid re-running it
def ensure_playable_video(input_path: str, output_path: str):
    subprocess.run([
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ], check=True)

st.title("Video Processor")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(uploaded_file.read())
        input_path = input_tmp.name

    raw_output = input_path.replace(".mp4", "_raw.mp4")
    final_output = input_path.replace(".mp4", "_processed.mp4")

    with st.spinner("Processing video..."):
        test(input_path, raw_output)
        ensure_playable_video(input_path, final_output)

    st.success("Processing complete!")

    # Read binary and display using Streamlit
    with open(final_output, 'rb') as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Save for cleanup
    if "cleanup" not in st.session_state:
        st.session_state.cleanup = []
    st.session_state.cleanup.extend([input_path, raw_output, final_output])

# Cleanup old temp files
for f in st.session_state.get("cleanup", []):
    try:
        os.remove(f)
    except OSError:
        pass
st.session_state.cleanup = []

# =================================================================
# To run it: streamlit run streamlit.py
# ================================================================= 