import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('data/recording_3/frames/frame_3 (80).png', cv2.IMREAD_GRAYSCALE)

# Resize the image to a smaller size 
scale_percent = 50  # Percentage of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# cv2.imshow("original image",image)

# ======= PREPROCESSING =======         prova ad aggiungerlo se ottieni brutti risultati

# # Apply Gaussian Blur to reduce noise and improve edge detection
# # Parameters: (source image, kernel size, standard deviation in X direction)
# blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# # Apply adaptive histogram equalization to improve contrast
# # Parameters: (clip limit for contrast, tile grid size)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# equalized_image = clahe.apply(blurred_image)

# # Apply median blur to further reduce noise
# # Parameters: (source image, kernel size)
# preprocessed_image = cv2.medianBlur(equalized_image, 5)

# ======= OTSU THRESHOLD FOR EDGE DETECTION =======       good automatic way to find the threshold

# Compute Otsu's threshold 
otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Set lower and upper thresholds relative to Otsu's threshold
lower = 0.5 * otsu_thresh
upper = 1.5 * otsu_thresh

edges = cv2.Canny(image, lower, upper)

# cv2.imshow("Canny Otsu", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Display the original image and the edge-detected image
# plt.subplot(121), plt.imshow(image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()



# Use Hough Line Transform to detect lines in the edge-detected image
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
# Draw the lines on the original image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the original image with detected lines
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image with Lines'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

# linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 50, 10)

# # Draw the lines on the original image
# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", edges)

