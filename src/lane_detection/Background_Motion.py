import cv2
import numpy as np


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
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Use RANSAC to filter out moving objects
            if len(src_pts) >= 10: 
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
