import os
from collections import deque
from statistics import mode

import pykitti
import numpy as np

import preparation.dataset as prepdata
from projection.project import points2image_project, frame2frame_project
from visualisation.image import show_image, plot_point_on_image
from prediction.predict import Predictor 

# The Path to the downloaded odometry file
pardir = r"/mnt"
calibration = os.path.join(pardir, "calib")
color = os.path.join(pardir, "color")
velodyne = os.path.join(pardir, "velodyne")
semantic = os.path.join(pardir, "semantic")

# The path of the temporary dateset
dataset = os.path.join(os.path.abspath(os.path.curdir), "dataset")
sequence = 0

# The path to the pretrained image semantic segmentation model
pretrained_model = r"/mnt/autodriving/prediction/kitti_best.pth"

# The path of the output dir
output_dir = r"/mnt/test/sequences" + f"/{sequence:02d}"


# co-predict image number
img_count = 5


if not os.path.exists(dataset):
    prepdata.generate(dataset, 
        calibration=calibration, 
        color=color, 
        velodyne=velodyne, 
        semantic=semantic,
    )

seq = pykitti.odometry(dataset, f"{sequence:02d}")
predictor = Predictor(pretrained_model)


index = 0
previous_img_labels= deque(maxlen=img_count)
previous_poses = deque(maxlen=img_count)

for velo, img, pose in zip(seq.velo, seq.cam2, seq.poses):
    pred = {
       tuple(velo[i][:3]) : [] for i in range(len(velo))
    }

    # current scan
    velo_xyz, img_xy = points2image_project(velo, seq.calib)
    img_labels = predictor.predict(img)

    for i in range(velo_xyz.shape[1]):

        try:
            label = img_labels[np.int32(img_xy[1][i]), np.int32(img_xy[0][i])]
        except IndexError:
            continue

        xyz = (velo_xyz[0][i], velo_xyz[1][i], velo_xyz[2][i])
        pred[xyz].append(label)

    # previous scans
    for prelabel, prepose in zip(previous_img_labels, previous_poses):

        # project current points to previous coordinate system
        pre_velo = frame2frame_project(velo, pose, prepose, seq.calib.T_cam0_velo)
        paired_points = { tuple(pre[:3]) : tuple(cur[:3]) for pre, cur in zip(pre_velo, velo) }

        # match previous points with previous image
        pre_velo_xyz, pre_img_xy = points2image_project(pre_velo, seq.calib)

        for i in range(pre_velo_xyz.shape[1]):
            try:
                label = prelabel[np.int32(pre_img_xy[1][i]), np.int32(pre_img_xy[0][i])]
            except IndexError:
                continue

            pre_xyz = (pre_velo_xyz[0][i], pre_velo_xyz[1][i], pre_velo_xyz[2][i])
            xyz = paired_points[pre_xyz]
            
            pred[xyz].append(label)
    
    # process to get final prediction
    final_pred = []

    for point in velo:
        xyz = (point[0], point[1], point[2])
        value = pred[xyz]

        if len(value) == 0:
            final_pred.append(0)
            continue
        final_pred.append(mode(value))

    final_pred = np.array(final_pred, dtype=np.int32)
    final_pred.tofile(output_dir + f"/{index:06d}.label") # TODO output to file
    print("Finish " + str(index))

    previous_img_labels.append(img_labels)
    previous_poses.append(pose)
    index += 1