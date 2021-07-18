import os
import pykitti
import cv2
import numpy as np

from preparation.dataset import generate, delete
from projection.project import points2image_project, frame2frame_project
from visualisation.image import show_image, plot_point_on_image

# The Path to the downloaded odometry file
pardir = r"C:\Users\timkh\Documents\Development\dataset\kitti"
calibration = os.path.join(pardir, "calib")
color = os.path.join(pardir, "color")
velodyne = os.path.join(pardir, "velodyne")
semantic = os.path.join(pardir, "semantic")

# The path of the dateset
dataset = os.path.join(os.path.abspath(os.path.curdir), "dataset")

# Field of view
v_fov = (-24.9, 2.0)
h_fov = (-45, 45)

if __name__ == '__main__':
    if not os.path.exists(dataset):
        generate(dataset, calibration=calibration, color=color, velodyne=velodyne, semantic=semantic)

    seq = pykitti.odometry(dataset, "00")
    index = 5

    for i in range(10):
        points = frame2frame_project(seq.get_velo(i), seq.poses[i], seq.poses[5], seq.calib.T_cam0_velo)
        img = np.array(seq.get_cam2(index))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        xyz_selected, xy_projected = points2image_project(points, seq.calib, v_fov, h_fov)

        show_image("Projected Image", plot_point_on_image(img, xyz_selected, xy_projected))

