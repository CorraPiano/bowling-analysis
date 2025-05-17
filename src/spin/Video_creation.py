import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter


# ==============================================================================
#                                   VIDEO FUNCTIONS
# ==============================================================================

'''Compute the rotation matrix for a given axis and angle'''
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([
        [a*a + b*b - c*c - d*d,     2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),     a*a + c*c - b*b - d*d,     2*(c*d - a*b)],
        [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
    ])

''' Compute the rotation matrix to align the north pole of the sphere to the current axis '''
def align_north_pole_to_vector(target_vector):
    target_vector = target_vector / np.linalg.norm(target_vector)
    north_pole = np.array([0, 0, 1])
    
    if np.allclose(target_vector, north_pole):
        return np.eye(3)
    
    if np.allclose(target_vector, -north_pole):
        # 180 degree rotation around any perpendicular axis
        return rotation_matrix(np.array([1, 0, 0]), np.pi)
    
    axis = np.cross(north_pole, target_vector)
    angle = np.arccos(np.dot(north_pole, target_vector))
    return rotation_matrix(axis, angle)

''' Create the 3d video of the sphere rotating around the axis from the dataframe '''
def create_sphere_video(df, output_path, fps=30):
    # video dimensions
    # final video size is 1920x1080 -> make it 300x300 instead of 900x900
    dpi = 150 # for real video 75 - for visualization 150
    figure_size = 6  # for real video 4 - for visualization 6

    # scaling factor for better visualization
    scaling_factor = 1.5
    # Sphere geometry
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x0 = np.cos(u) * np.sin(v) * scaling_factor
    y0 = np.sin(u) * np.sin(v) * scaling_factor
    z0 = np.cos(v) * scaling_factor

    # Create checkerboard pattern
    colors = np.empty(u.shape, dtype=object)

    # Number of bands of the ball (latitude and longitude)
    num_lat_bands = 1
    num_lon_bands = 4

    # # Colors
    # band_colors = [(69/255, 24/255, 91/255, 1.0),  # violet
    #                (57/255, 148/255, 160/255, 1.0)]  # light-blue
    band_colors = [(0/255, 0/255, 0/255, 1.0), 
                (255/255, 255/255, 255/255, 1.0)]  
    highlight_color = (1.0, 0.0, 0.0, 1.0)  # red


    # Thickness values
    lat_step = np.pi / 50
    equator_thickness = 2 * lat_step  # about two latitude steps
    pole_thickness = lat_step     # about two latitude steps

    # Equator bounds
    equator_lower = (np.pi / 2) - equator_thickness / 2
    equator_upper = (np.pi / 2) + equator_thickness / 2

    # North and South pole bounds
    north_pole_upper = pole_thickness
    south_pole_lower = np.pi - pole_thickness

    # Assign colors
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            v_val = v[i, j]
            if (equator_lower <= v_val <= equator_upper or
                v_val <= north_pole_upper or
                v_val >= south_pole_lower):
                colors[i, j] = highlight_color
            else:
                lat_index = int(v_val // (np.pi / num_lat_bands))
                lon_index = int(u[i, j] // (2 * np.pi / num_lon_bands))
                colors[i, j] = band_colors[(lat_index + lon_index) % 2]

    # Initialize plot
    fig = plt.figure(figsize=(figure_size, figure_size))
    fig.patch.set_facecolor((255/255, 219/255, 133/255)) # Set background color to light brown
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((255/255, 219/255, 133/255)) # Set background color to light brown
    metadata = dict(title='Rotating Sphere with Axis DF')
    writer = FFMpegWriter(fps=fps, metadata=metadata)    # Set the desired FPS

    first_valid_index = df['x_axis'].first_valid_index()
    # print(f"First valid index where 'x_axis' is not NaN: {first_valid_index}")

    last_valid_index = df['x_axis'].last_valid_index()
    # print(f"Last valid index where 'x_axis' is not NaN: {last_valid_index}")

    # compute the average angle to assign it whene angle is NaN
    mean_angle = df['angle'].mean()

    # Initialize variables
    R_total = np.eye(3)
    dtheta = 0
    angular_velocity = np.nan

    # Create and save video
    with writer.saving(fig, output_path, dpi=dpi):
        for i, row in df.iterrows():
            ax.cla()

            # CASE: before the first valid data for the axis
            if np.isnan(row['x_axis']) and i <= first_valid_index:
                # Align sphere to the axis
                point = np.array([df.loc[first_valid_index]['x_axis'], df.loc[first_valid_index]['y_axis'], df.loc[first_valid_index]['z_axis']])
                angular_velocity = df.loc[first_valid_index]['angle'] * fps

                R_align = align_north_pole_to_vector(point)
                # Apply to sphere coordinates once
                x = R_align[0, 0]*x0 + R_align[0, 1]*y0 + R_align[0, 2]*z0
                y = R_align[1, 0]*x0 + R_align[1, 1]*y0 + R_align[1, 2]*z0
                z = R_align[2, 0]*x0 + R_align[2, 1]*y0 + R_align[2, 2]*z0


                # Draw an arrow to indicate the angular velocity
                origin = -point * 1.6
                vector = -point * abs(angular_velocity) * 0.05  # Adjust the scaling factor as needed
                # ax.quiver(*origin, *vector, color='black', linewidth=2)

                # Draw a thick line representing the vector
                ax.plot([origin[0], vector[0]+origin[0]], [origin[1], vector[1]+origin[1]], [origin[2], vector[2]+origin[2]],
                        color='red', linewidth=4, zorder=10)  # zorder might help
                # Add a scatter point (like an arrowhead)
                ax.scatter([vector[0]+origin[0]], [vector[1]+origin[1]], [vector[2]+origin[2]],
                        color='red', s=60, marker='o', zorder=10)

                # Draw the sphere
                ax.plot_surface(x, y, z, facecolors=colors, edgecolor='gray', alpha=1, linewidth=0.2)


                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                ax.view_init(elev=-90, azim=-90)
                ax.axis('off')

                # Draw the text
                ax.text2D(0.0, 0.09, 'N/D rad/s', ha='center', va='center', fontsize=30, color='black',
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3', linewidth=2))

                writer.grab_frame()
                # break
                continue

            # CASE: after the last valid data for the axis
            if np.isnan(row['x_axis']) and i >= last_valid_index:
                # Align sphere to the axis
                point = np.array([df.loc[last_valid_index]['x_axis'], df.loc[last_valid_index]['y_axis'], df.loc[last_valid_index]['z_axis']])
                angular_velocity = df.loc[last_valid_index]['angle'] * fps
                R_align = align_north_pole_to_vector(point)
                # Apply to sphere coordinates once
                x = R_align[0, 0]*x0 + R_align[0, 1]*y0 + R_align[0, 2]*z0
                y = R_align[1, 0]*x0 + R_align[1, 1]*y0 + R_align[1, 2]*z0
                z = R_align[2, 0]*x0 + R_align[2, 1]*y0 + R_align[2, 2]*z0

                # Draw the sphere
                ax.plot_surface(x, y, z, facecolors=colors, edgecolor='gray', alpha=1, linewidth=0.2)


                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                ax.view_init(elev=-90, azim=-90)
                ax.axis('off')

                # Draw the text
                ax.text2D(0.0, 0.09, 'N/D rad/s', ha='center', va='center', fontsize=30, color='black',
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3', linewidth=2))


                # Draw an arrow to indicate the angular velocity
                origin = -point * 1.6
                vector = -point * abs(angular_velocity) * 0.05  # Adjust the scaling factor as needed
                # ax.quiver(*origin, *vector, color='black', linewidth=2)


                # Draw a thick line representing the vector
                ax.plot([origin[0], vector[0]+origin[0]], [origin[1], vector[1]+origin[1]], [origin[2], vector[2]+origin[2]],
                        color='red', linewidth=4, zorder=10)  # zorder might help
                # Add a scatter point (like an arrowhead)
                ax.scatter([vector[0]+origin[0]], [vector[1]+origin[1]], [vector[2]+origin[2]],
                        color='red', s=60, marker='o', zorder=10)

                writer.grab_frame()
                continue

            # Get axis
            axis = row[['x_axis', 'y_axis', 'z_axis']].values

            if np.linalg.norm(axis) < 1e-6:
                continue

            # set new pole direction
            point = np.array([row['x_axis'], row['y_axis'], row['z_axis']])
            R_align = align_north_pole_to_vector(point)

            # compute the new position o the sphere (rotation matrix)
            if np.isnan(row['angle']):
                # If angle is NaN, use the previous angle
                dtheta -= mean_angle
            else:
                dtheta -= row['angle']
            R_spin = rotation_matrix(np.array([0, 0, 1]), dtheta)
            R_total = R_align @ R_spin

            # Apply to sphere coordinates once
            x = R_total[0, 0]*x0 + R_total[0, 1]*y0 + R_total[0, 2]*z0
            y = R_total[1, 0]*x0 + R_total[1, 1]*y0 + R_total[1, 2]*z0
            z = R_total[2, 0]*x0 + R_total[2, 1]*y0 + R_total[2, 2]*z0

            # Draw the sphere
            ax.plot_surface(x, y, z, facecolors=colors, edgecolor='gray', alpha=1, linewidth=0.01)

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])

            ax.view_init(elev=-90, azim=-90)
            ax.axis('off')

            # Calculate angular velocity and display it
            angular_velocity = row['angle'] * fps

            # Display text
            if not np.isnan(angular_velocity):
                if angular_velocity > 0:
                    ax.text2D(0.0, 0.09, f'{angular_velocity:.2f}rad/s ↑', ha='center', va='center', fontsize=30, color='black',
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3', linewidth=2))
                else:
                    ax.text2D(0.0, 0.09, f'{angular_velocity:.2f}rad/s ↓', ha='center', va='center', fontsize=30, color='black',
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3', linewidth=2))

            # Draw an arrow to indicate the angular velocity
            origin = -point * 1.6
            vector = -point * abs(angular_velocity) * 0.05  # Adjust the scaling factor as needed
            # ax.quiver(*origin, *vector, color='black', linewidth=2)


            # Draw a thick line representing the vector
            ax.plot([origin[0], vector[0]+origin[0]], [origin[1], vector[1]+origin[1]], [origin[2], vector[2]+origin[2]],
                    color='red', linewidth=4, zorder=10)
            # Add a scatter point (like an arrowhead)
            ax.scatter([vector[0]+origin[0]], [vector[1]+origin[1]], [vector[2]+origin[2]],
                    color='red', s=60, marker='o', zorder=10)


            writer.grab_frame()
        print(f"Video saved to {output_path}")

''' Crop the video '''
def crop_video(video_path, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    FPS = round(cap.get(cv2.CAP_PROP_FPS))

    # Read the Dimensions of the first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Non è stato possibile leggere il primo frame.")

    h, w, _ = frame.shape

    # Define margins for cropping
    crop_top = 0
    crop_bottom = 200
    crop_left = 80
    crop_right = 80

    # Nuove dimensioni
    new_h = h - crop_top - crop_bottom
    new_w = w - crop_left - crop_right

    # new video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (new_w, new_h))

    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #  Loop through the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fine del video.")
            break

        # Crop the frame
        cropped = frame[crop_top:h - crop_bottom, crop_left:w - crop_right]

        # write the cropped frame to the new video
        out.write(cropped)

    # Rilascio risorse
    cap.release()
    out.release()
    print(f"Video salvato in: {output_path}")

# ==============================================================================
#                              OVERVIEW FUNCTION
# ==============================================================================

''' Main function to detect the spin, process the data and create the sphere video '''
def spin_video_creation(video_path, sphere_path, output_path, input_data_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_data_path)
    
    create_sphere_video(df, sphere_path, fps=fps)

    # crop the video
    crop_video(sphere_path, output_path)

    # Release resources
    cap.release()