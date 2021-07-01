import os


def generate(path, calibration, color=None, velodyne=None, semantic=None):
    sequence_path = os.path.join(path, "sequences")
    os.mkdir(path)
    os.mkdir(sequence_path)

    _load_calib(sequence_path, calibration)

    if color is not None:
        _load_color(sequence_path, color)

    if velodyne is not None:
        _load_velodyne(sequence_path, velodyne)

    if semantic is not None:
        _load_semantic(sequence_path, semantic)


def delete(path):
    # delete sequence
    sequence_path = os.path.join(path, "sequences")
    if os.path.exists(sequence_path):
        with os.scandir(sequence_path) as entries:
            for entry in entries:
                entry_path = os.path.join(sequence_path, entry.name)

                with os.scandir(entry_path) as symlinks:
                    for link in symlinks:
                        os.unlink(link.path)

                os.rmdir(entry_path)
        os.rmdir(sequence_path)

    # delete poses
    poses_path = os.path.join(path, "poses")
    if os.path.exists(poses_path):
        with os.scandir(poses_path) as entries:
            for entry in entries:
                os.unlink(entry.path)
        os.rmdir(poses_path)

    os.rmdir(path)


def _load_calib(sequence_path, calibration):
    with os.scandir(os.path.join(calibration, "sequences")) as entries:
        for entry in entries:  # for each sequence
            entry_path = os.path.join(sequence_path, entry.name)
            os.mkdir(entry_path)

            with os.scandir(entry.path) as files:
                for file in files:
                    source = os.path.abspath(file.path)
                    target = os.path.join(entry_path, file.name)
                    os.symlink(source, target)


def _load_color(sequence_path, color):
    with os.scandir(os.path.join(color, "sequences")) as entries:
        for entry in entries:
            entry_path = os.path.join(sequence_path, entry.name)

            image_2_source = os.path.abspath(os.path.join(entry.path, "image_2"))
            image_2_target = os.path.join(entry_path, "image_2")
            os.symlink(image_2_source, image_2_target)

            image_3_source = os.path.abspath(os.path.join(entry.path, "image_3"))
            image_3_target = os.path.join(entry_path, "image_3")
            os.symlink(image_3_source, image_3_target)


def _load_velodyne(sequence_path, velodyne):
    with os.scandir(os.path.join(velodyne, "sequences")) as entries:
        for entry in entries:
            entry_path = os.path.join(sequence_path, entry.name)

            source = os.path.abspath(os.path.join(entry.path, "velodyne"))
            target = os.path.join(entry_path, "velodyne")
            os.symlink(source, target)


def _load_semantic(sequence_path, semantic):
    poses_path = os.path.join(sequence_path, os.pardir, "poses")
    os.mkdir(poses_path)

    with os.scandir(os.path.join(semantic, "sequences")) as entries:
        for entry in entries:
            entry_path = os.path.join(sequence_path, entry.name)

            label_source = os.path.abspath(os.path.join(entry.path, "labels"))
            label_target = os.path.join(entry_path, "labels")
            os.symlink(label_source, label_target)

            poses_source = os.path.abspath(os.path.join(entry.path, "poses.txt"))
            poses_target = os.path.join(entry_path, "poses.txt")
            poses_folder_target = os.path.join(poses_path, "{}.txt".format(entry.name))
            os.symlink(poses_source, poses_target)
            os.symlink(poses_source, poses_folder_target)


if __name__ == '__main__':
    calib_file = os.path.join(os.curdir, "dataset", "calibration")
    color_file = os.path.join(os.curdir, "dataset", "color")
    velodyne_file = os.path.join(os.curdir, "dataset", "velodyne")
    semantic_file = os.path.join(os.curdir, "dataset", "semantic")

    # generate("dataset", calib_file, color=color_file, velodyne=velodyne_file, semantic=semantic_file)
    delete("dataset")
