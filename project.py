import numpy as np


# Field of view
v_fov = (-24.9, 2.0)
h_fov = (-45, 45)


def _in_view_points(origin_points):
    x = origin_points[:, 0]
    y = origin_points[:, 1]
    z = origin_points[:, 2]

    h_in_range = np.logical_and(np.arctan2(y, x) < (h_fov[1] * np.pi / 180),
                                np.arctan2(y, x) > (h_fov[0] * np.pi / 180))
    v_in_range = np.logical_and(np.arctan2(z, x) < (v_fov[1] * np.pi / 180),
                                np.arctan2(z, x) > (v_fov[0] * np.pi / 180))
    mask = np.logical_and(h_in_range, v_in_range)

    x_in = x[mask]
    y_in = y[mask]
    z_in = z[mask]
    xyz = np.vstack((x_in, y_in, z_in))

    one = np.full((1, xyz.shape[1]), 1)
    xyz = np.vstack((xyz, one))

    return xyz


def points2image_project(points, calib):
    """" project points in velo onto the image

    :param points: The velodyne data
    :param calib: The sequence calibration data

    :return velo, image: the in range points and its coordinates on image
    """
    # Transformation Matrix
    velo_to_camera = calib.T_cam2_velo
    camera_to_image = calib.K_cam2

    velo = _in_view_points(points)

    camera = velo_to_camera @ velo
    camera = np.delete(camera, 3, axis=0)

    image = camera_to_image @ camera
    image = image[::] / image[::][2]
    image = np.delete(image, 2, axis=0)

    return velo, image


def frame2frame_project(frame, pose, target_pose, tr):
    """" project points from one frame onto the coordinate system of another frame

    :param frame: the velodyne data of the frame
    :param pose: the pose data of the current frame
    :param target_pose: the pose data of the frame whose coordinate system is used
    :param tr: the sequence calibration data Tr or T_cam0_velo

    :return projected_points: the points in the target coordinate system
    """
    # save the remission and copy the points coordinate
    remissions = frame[:, 3]
    points = np.ones(frame.shape)
    points[:, 0:3] = frame[:, 0:3]

    # calculate transform matrix
    tr_inv = np.linalg.inv(tr)
    pose = tr_inv @ pose @ tr
    target_pose = tr_inv @ target_pose @ tr
    diff = np.linalg.inv(target_pose) @ pose

    projected_points = ( diff @ points.T ).T
    projected_points[:, 3] = remissions

    return projected_points

