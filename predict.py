import numpy as np
from statistics import mode

from project import points2image_project, frame2frame_project


def predit_on_label(
    point,  # current frame point clouds
    image_label,    # current frame cam2 label
    pose,   # current frame pose
    previous_labels,    # previous cam2 labels to look at
    previous_poses,  # previous_label correspoding frame pose
    calib,  # sequence calibration
):
    prediction = {
        tuple(p[:3]): [] for p in point
    }

    velo_xzy, img_xy = points2image_project(point, calib)

    for i in range(velo_xzy.shape[1]):
        try:
            label = image_label[np.int32(img_xy[1][i]), np.int32(img_xy[0][i])]
        except IndexError:
            # This indicates this point is out of view in the image
            continue

        pred_xyz = (velo_xzy[0][i], velo_xzy[1][i], velo_xzy[2][i])
        prediction[pred_xyz].append(label)

    for pl, pp in zip(previous_labels, previous_poses):
        # project current point to previous frames
        point_in_previous = frame2frame_project(
            point, pose, pp, calib.T_cam0_velo)
        paired_points = {
            tuple(p[:3]): tuple(c[:3]) for p, c in zip(point_in_previous, point)
        }

        # match previous points with previous image label
        pre_xyz, pre_xy = points2image_project(point_in_previous, calib)

        for i in range(pre_xyz.shape[1]):
            try:
                label = pl[np.int32(pre_xy[1][i]), np.int32(pre_xy[0][i])]
            except IndexError:
                continue

            prediction[paired_points[
                (pre_xyz[0][i], pre_xyz[1][i], pre_xyz[2][i])
            ]].append(label)

    final_pred = []
    for p in point:
        preds = prediction[(p[0], p[1], p[2])]

        if len(preds) == 0:
            final_pred.append(0)
            continue
        final_pred.append(mode(preds))
    final_pred = np.array(final_pred, dtype=np.int32)

    return final_pred
