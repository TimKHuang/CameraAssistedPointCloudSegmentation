import numpy as np
import os
from statistics import mode
from deepwv3plus_net import DeepWV3Plus, img_transform
from feature_net import FeatureNet
import torch

from project import points2image_project, frame2frame_project
from kitti_labels import kitti_learning2index


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

def predict_on_feature(
    kitti,
    weights_addr,
    num_prev=4):
    
    net = DeepWV3Plus(is_feature_extractor=True)
    net = torch.nn.DataParallel(net).cuda()
    print('Image Net Built')

    net_state_dict = net.state_dict()
    loaded_dict = torch.load(weights_addr, map_location=torch.device('cpu'))
    loaded_dict = loaded_dict['state_dict']
    updated_dict = {}

    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            updated_dict[k] = loaded_dict[k]

    net.load_state_dict(updated_dict)
    net.eval()
    print('Image Net restored')

    feature_net = FeatureNet().cuda()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    feature_net.load_state_dict(torch.load("./feature_net.pth"))
    feature_net.eval()
    print('Feature Net restored')

    all_predictions = []
    for index in range(len(kitti)):
        velo = kitti.get_velo(index)
        feature_dict = {
            tuple(v[:3]): [] for v in velo
        }

        # Same frame projection
        cur_xyz, cur_xy = points2image_project(velo, kitti.calib)

        img = kitti.get_cam2(index)
        with torch.no_grad():
            img_tensor = img_transform(img)

            img_feature = net(img_tensor.unsqueeze(0).cuda())
            img_feature = img_feature.cpu().numpy().squeeze()

        for i in range(cur_xyz.shape[1]):
            try:
                feature = img_feature[
                    :,
                    np.int32(cur_xy[1][i]),
                    np.int32(cur_xy[0][i])
                ]
            except IndexError:
                continue

            key = (
                cur_xyz[0][i],
                cur_xyz[1][i],
                cur_xyz[2][i]
            )
            feature_dict[key].append(feature)

        # Previous Frame Projection
        for p in range(1, num_prev):
            if (index - p ) < 0:
                continue 
            pre_xyz = frame2frame_project(
                velo,
                kitti.poses[index],
                kitti.poses[index - p],
                kitti.calib.T_cam0_velo
            )
            points_dict = {
                tuple(pre[:3]): tuple(cur[:3])
                for pre, cur in zip(pre_xyz, velo)
            }

            pre_xyz, pre_xy = points2image_project(pre_xyz, kitti.calib)

            img = kitti.get_cam2(index - p)
            with torch.no_grad():
                img_tensor = img_transform(img)

                img_feature = net(img_tensor.unsqueeze(0).cuda())
                img_feature = img_feature.cpu().numpy().squeeze()

            for i in range(pre_xyz.shape[1]):
                try:
                    feature = img_feature[
                        :,
                        np.int32(pre_xy[1][i]),
                        np.int32(pre_xy[0][i])
                    ]
                except IndexError:
                    continue

                key = points_dict[(
                    pre_xyz[0][i],
                    pre_xyz[1][i],
                    pre_xyz[2][i]
                )]
                feature_dict[key].append(feature)

        final_pred = []
        for v in velo:
            key = tuple(v[:3])
            f = feature_dict[key]

            predict = -1
            # It has not been mapped to any image
            if len(f) == num_prev:
                feature = np.mean(f, axis=0)
                feature = torch.tensor(feature, dtype=torch.float)
                with torch.no_grad():
                    predict = feature_net(feature.unsqueeze(0).cuda())
                    predict = predict.cpu().numpy().squeeze()
                predict = kitti_learning2index[np.argmax(predict)] 
            
            if predict == -1:
                final_pred.append(0)
            else:
                final_pred.append(predict)
        
        final_pred = np.array(final_pred, dtype=np.int32)
        all_predictions.append(final_pred)
        print("finish " + str(index))
    
    return all_predictions


