import os
import pykitti
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from deepwv3plus_net import DeepWV3Plus, img_transform
from project import points2image_project, frame2frame_project
from kitti_labels import kitti_label2learning


def generate_dataset(
        sequence,
        save_dir,
        kitti_dir,
        weights_addr,
        dataset_size=10000,
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
        for p in range(1, num_prev):
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
            if len(f) != num_prev:
                continue

            feature = np.mean(f, axis=0)
            feature = np.append(feature, np.float32(label_dict[k]))
            np.save(os.path.join(save_dir, f"{current_datasize:08d}"), feature)

            current_datasize += 1
            if current_datasize % (dataset_size // 5) == 0:
                print(f"{current_datasize} / {dataset_size} data added.")

    print(f"{current_datasize} / {dataset_size} data added.")


class FeatureDB(Dataset):

    def __init__(self, dataset_dir) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(os.listdir(self.dataset_dir))

    def __getitem__(self, index):
        raw = np.load(os.path.join(self.dataset_dir, f"{index:08d}.npy"))
        features = raw[:-1]
        label = int(raw[-1]) & 0xFFFF
        label = kitti_label2learning[label]
        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(label, dtype=torch.long)
        )
