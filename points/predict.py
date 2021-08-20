from collections import deque
from statistics import mode
import numpy as np

from points.project import frame2frame_project
from points.project import points2image_project


def point_predit(seq, predictor, prev_count):
    """ pred the velo in a sequnce

    :param seq: the pykitti.odometry sequence
    :param predictor: the imgPredictor
    :param prev_count: count on how many previous image to compare

    :yield: the result of each scan
    """

    previous_img_labels = deque(maxlen=prev_count)
    previous_poses = deque(maxlen=prev_count)

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
        
        previous_img_labels.append(img_labels)
        previous_poses.append(pose)

        yield final_pred
    