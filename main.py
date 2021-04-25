import os
import cv2

from preparation.odometryDataset import Dataset

# The Path to the downloaded odometry file
calibration = os.path.join(os.path.pardir, "dataset", "calibration")
color = os.path.join(os.path.pardir, "dataset", "color")
velodyne = os.path.join(os.path.pardir, "dataset", "velodyne")

if __name__ == '__main__':
    pass
