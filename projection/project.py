import cv2
import numpy as np
from pykitti import odometry


def in_view_points(origin_points, v_fov, h_fov):
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
    print("in range points size: " + str(xyz.shape))

    one = np.full((1, xyz.shape[1]), 1)
    xyz = np.vstack((xyz, one))

    return xyz


def points_to_image_project(sequence, index, v_fov, h_fov):
    """
    Project from 3D points of velodyne to 2D image of camera in the indexed place in the sequence
    :param odometry sequence: the sequence of dataset
    :param int index: the index of the pair of camera and velodyne
    :param (float, float) v_fov: vertical field of view of the camera
    :param (float, float) h_fov: horizontal field of view of thr camera
    :return: origin coordinates, projected points coordinates and the corresponding color representing distance
    """
    origin_points = sequence.get_velo(index)

    # Transformation Matrix
    velo_to_camera = sequence.calib.T_cam2_velo
    camera_to_image = sequence.calib.K_cam2

    xyz_velo = in_view_points(origin_points, v_fov, h_fov)

    xyz_camera = np.matmul(velo_to_camera, xyz_velo)
    xyz_camera = np.delete(xyz_camera, 3, axis=0)

    xy_image = np.matmul(camera_to_image, xyz_camera)
    xy_image = xy_image[::] / xy_image[::][2]
    xy_image = np.delete(xy_image, 2, axis=0)

    return xyz_velo, xy_image
