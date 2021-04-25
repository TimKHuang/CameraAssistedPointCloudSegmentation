import cv2
import numpy as np
from pykitti import odometry


def depth_color(value, minimum=0, maximum=120):
    value = np.clip(value, minimum, maximum)
    return ((value - minimum) / (maximum - minimum) * 120).astype(np.uint8)


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

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dist_lim = dist[mask]
    xyz_color = depth_color(dist_lim, 0, 70)
    return xyz, xyz_color


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

    xyz_velo, xyz_color = in_view_points(origin_points, v_fov, h_fov)

    xyz_camera = np.matmul(velo_to_camera, xyz_velo)
    xyz_camera = np.delete(xyz_camera, 3, axis=0)

    xyz_camera = np.matmul(camera_to_image, xyz_camera)

    xy_image = xyz_camera[::] / xyz_camera[::][2]
    xy_image = np.delete(xy_image, 2, axis=0)

    return xyz_velo, xy_image, xyz_color


if __name__ == '__main__':
    import os
    from preparation.odometryDataset import Dataset

    calib = os.path.join(os.path.pardir, os.path.pardir, "dataset", "calibration")
    color = os.path.join(os.path.pardir, os.path.pardir, "dataset", "color")
    velodyne = os.path.join(os.path.pardir, os.path.pardir, "dataset", "velodyne")

    with Dataset.open_odometry(calib, color=color, velodyne=velodyne) as dataset:
        seq = dataset.get_sequence("00")
        points, color = points_to_image_project(seq, 0, (-24.9, 2.0), (-45, 45))
        print("COLOR!")
        print(color)

        img_np = np.array(seq.get_cam2(0))
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imshow("img", img_cv2)
        cv2.waitKey(0)

        hsv_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

        final = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.imshow("img", final)
        cv2.waitKey(0)

        print("SUCCESS")
