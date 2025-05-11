from pathlib import Path
import subprocess
import os

def get_video_height(video_path):
    # Use ffprobe to get video dimensions
    command = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'stream=height', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    height = int(result.stdout.decode().strip())
    return height

def create_final_video(input_video_path, input_video_lane, input_video_sphere, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Get the height of the first video (left video)
    left_video_height = get_video_height(input_video_path)

    # Step 2: Resize the second video (right video) to match the height of the first video
    # Construct the ffmpeg command for resizing the second video
    resized_right_video = "resized_right_video.mp4"
    padded_right_video = "padded_right_video.mp4"
    resize_command = [
        'ffmpeg', 
        '-i', input_video_lane, 
        '-vf', f"scale=-1:{left_video_height}",  # Preserve aspect ratio and set height
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '18', 
        resized_right_video
    ]
    subprocess.run(resize_command, check=True)

    # Add a 5-pixel black border to the left
    pad_command = [
        "ffmpeg",
        "-i", str(resized_right_video),
        "-vf", "pad=width=iw+5:height=ih:x=5:y=0:color=black",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        str(padded_right_video)
    ]
    subprocess.run(pad_command, check=True)

    # Step 3: Resize the sphere video to a smaller size (you can adjust the size here)
    resized_sphere_video = "resized_sphere_video.mp4"
    padded_sphere_video = "padded_sphere_video.mp4"
    sphere_width = 300  # Set the desired width for the sphere video
    resize_sphere_command = [
        'ffmpeg', 
        '-i', input_video_sphere, 
        '-vf', f"scale={sphere_width}:-1",  # Adjust width, keep aspect ratio
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '18', 
        resized_sphere_video
    ]
    subprocess.run(resize_sphere_command, check=True)

    # Add a 5-pixel black border around the entire video
    pad_command = [
        "ffmpeg",
        "-i", str(resized_sphere_video),
        "-vf", "pad=width=iw+5:height=ih+5:x=5:y=0:color=black",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        str(padded_sphere_video)
    ]
    subprocess.run(pad_command, check=True)

    # Step 4: Overlay the sphere video on the top-right corner of the left video
    # We will use ffmpeg's overlay filter to place the resized sphere video on top of the left video
    overlayed_left_video = "overlayed_left_video.mp4"
    overlay_command = [
        'ffmpeg', 
        '-i', input_video_path, 
        '-i', padded_sphere_video, 
        '-filter_complex', f"[0:v][1:v]overlay=W-w-0:0",  # Top-right corner, with 10px margin
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '18', 
        overlayed_left_video
    ]
    subprocess.run(overlay_command, check=True)

    # Step 5: Combine the overlayed left video and the resized right video side by side
    combine_command = [
        'ffmpeg', 
        '-i', overlayed_left_video, 
        '-i', padded_right_video, 
        '-filter_complex', 
        f"[0:v][1:v]hstack=inputs=2",  # Concatenate the videos side by side
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '18', 
        output_path
    ]
    subprocess.run(combine_command, check=True)

    # Clean up temporary files (optional)
    # Elimina i file se esistono
    for file_path in [resized_right_video, resized_sphere_video, overlayed_left_video, padded_sphere_video, padded_right_video]:
        if os.path.exists(file_path):
            os.remove(file_path)