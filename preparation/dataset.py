import os


def generate(path, calibration, color=None, velodyne=None, label=None):
    sequence_path = os.path.join(path, "sequences")
    os.mkdir(path)
    os.mkdir(sequence_path)

    _load_calib(sequence_path, calibration)

    if color is not None:
        _load_color(sequence_path, color)

    if velodyne is not None:
        _load_velodyne(sequence_path, velodyne)

    if label is not None:
        _load_label(sequence_path, label)


def delete(path):
    sequence_path = os.path.join(path, "sequences")
    with os.scandir(sequence_path) as entries:
        for entry in entries:
            entry_path = os.path.join(sequence_path, entry.name)

            with os.scandir(entry_path) as symlinks:
                for link in symlinks:
                    os.unlink(link.path)

            os.rmdir(entry_path)

    os.rmdir(sequence_path)
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


def _load_label(sequence_path, label):
    with os.scandir(os.path.join(label, "sequences")) as entries:
        for entry in entries:
            entry_path = os.path.join(sequence_path, entry.name)

            label_source = os.path.abspath(os.path.join(entry.path, "labels"))
            label_target = os.path.join(entry_path, "labels")
            os.symlink(label_source, label_target)

            poses_source = os.path.abspath(os.path.join(entry.path, "poses.txt"))
            poses_target = os.path.join(entry_path, "poses.txt")
            os.symlink(poses_source, poses_target)


if __name__ == '__main__':
    calib_file = os.path.join(os.path.pardir, os.path.pardir, "dataset", "calibration")
    color_file = os.path.join(os.path.pardir, os.path.pardir, "dataset", "color")
    velodyne_file = os.path.join(os.path.pardir, os.path.pardir, "dataset", "velodyne")
    label_file = os.path.join(os.path.pardir, os.path.pardir, "dataset", "label")

    # generate("dataset", calib_file, color=color_file, velodyne=velodyne_file, label=label_file)
    delete("dataset")
