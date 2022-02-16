from ast import Index
import os
from matplotlib.pyplot import axis
from sklearn.pipeline import FeatureUnion
import torch
import pykitti
import numpy as np
from collections import deque

from deepwv3plus_net import DeepWV3Plus, img_transform
from project import points2image_project, frame2frame_project


def generate_dataset(
        sequence,
        save_dir,
        kitti_dir,
        weights_addr,
        dataset_size=100000,
        num_prev=3):
    '''
    sequence: array like structure containing string of sequence.
    save_dir: the parent directory of sequences
    '''
    net = DeepWV3Plus(is_feature_extractor=True)
    net = torch.nn.DataParallel(net).cuda()
    print('Net Built')

    net_state_dict = net.state_dict()
    loaded_dict = torch.load(weights_addr, map_location=torch.device('cpu'))
    loaded_dict = loaded_dict['state_dict']
    updated_dict = {}

    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            updated_dict[k] = loaded_dict[k]

    net.load_state_dict(updated_dict)
    net.eval()
    print('Net restored')

    kitti_datasets = [
        pykitti.odometry(kitti_dir, sequence=seq) for seq in sequence
    ]

    current_datasize = 0
    project_times_count = np.zeros((num_prev + 1, ))
    while current_datasize < dataset_size:
        # Random sequence
        seq = np.random.randint(0, len(sequence))
        kitti = kitti_datasets[seq]
        # Random Frame
        index = np.random.randint(num_prev, len(kitti))
        # print(f"Generating datasets from {sequence[seq]} frame {index}.")

        # Read Label
        velo = kitti.get_velo(index)
        labels = np.fromfile(
            os.path.join(kitti_dir, 'sequences',
                         sequence[seq], 'labels', f'{index:06d}.label'),
            dtype=np.int32
        )
        label_dict = {
            tuple(v[:3]): l
            for v, l in zip(velo, labels)
        }

        # Generate Feature
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
        for p in range(num_prev):
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

        for k, f in feature_dict.items():
            # It has not been mapped to any image
            if len(f) == 0:
                continue

            # Everyone else has at least one projected feature
            project_times_count[len(f) - 1] += 1
            feature = np.mean(f, axis=0)
            np.append(feature, label_dict[k])
            np.save(os.path.join(save_dir, f"{current_datasize:08d}"), feature)

            current_datasize += 1
            if current_datasize % (dataset_size // 5) == 0:
                print(f"{current_datasize} / {dataset_size} data added.")
                print(f"Including {project_times_count} of different projections")

    print(f"{current_datasize} / {dataset_size} data added.")
    print(f"Including {project_times_count} of different projections")