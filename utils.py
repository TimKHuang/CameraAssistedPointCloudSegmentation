# This two function is used to get previous frame labels and poses to project.

def get_labels(labels, start_index, end_index):
    if start_index == end_index:
        return []

    return [
        labels[i]
        for i in range(start_index, end_index)
    ]


def get_poses(kitti_dataset, start_index, end_index):
    if start_index == end_index:
        return []

    return [
        kitti_dataset.poses[i]
        for i in range(start_index, end_index)
    ]
