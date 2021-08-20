import os
from collections import deque
from statistics import mode

import pykitti
import numpy as np

import files.dataset as dataset
from files.output import PredWriter
from img.predict import ImagePredictor 
from points.predict import point_predit


###################  Config  ################### 

# The Path to the downloaded odometry file
pardir = r"/mnt"
calibration = os.path.join(pardir, "calib")
color = os.path.join(pardir, "color")
velodyne = os.path.join(pardir, "velodyne")
semantic = os.path.join(pardir, "semantic")

# The path of the temporary dateset
dataset_dir = os.path.join(os.path.abspath(os.path.curdir), "dataset")
sequences = [0, 1] 

# The path to the pretrained image semantic segmentation model
pretrained_model = r"/mnt/autodriving/img/kitti_best.pth"

# The path of the output dir
output_dir = r"/mnt/result"

# co-predict image number
prev_count = 5

################################################ 

dataset.generate(
    dataset_dir, 
    calibration=calibration, 
    color=color, 
    velodyne=velodyne, 
    semantic=semantic,
)
print("Dataset symlinks generated")

predictor = ImagePredictor(pretrained_model)
writer = PredWriter(output_dir)
writer.init_path()

for s in sequences:
    writer.switch_sequence(s)
    seq = pykitti.odometry(dataset_dir, f"{s:02d}")

    for scan_result in point_predit(seq, predictor, prev_count):
        writer.write(scan_result)
    
    print("Sequence " + str(s) + "predictions finished")

dataset.delete(dataset_dir)
print("temporary links deleted")