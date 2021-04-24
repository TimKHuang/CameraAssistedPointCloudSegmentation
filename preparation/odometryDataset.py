import pykitti
import os
import shutil


class Dataset:

    def __init__(self, store_path = None):
        if store_path is None:
            store_path = os.path.curdir
        self.path = os.path.join(store_path, "temp")
        os.mkdir(self.path)

        self.sequence_path = os.path.join(self.path, "sequences")
        os.mkdir(self.sequence_path)

    def prepare_odometry(self, calibration, grayscale=None, color=None, velodyne=None):
        self._load_calib(calibration)

        if grayscale is not None:
            self._load_grayscale(grayscale)

        if color is not None:
            self._load_color(color)

        if velodyne is not None:
            self._load_velodyne(velodyne)

    def get_odometry(self, sequence):
        return pykitti.odometry(self.path, sequence)

    def close(self):
        with os.scandir(self.sequence_path) as entries:
            for entry in entries:
                entry_path = os.path.join(self.sequence_path, entry.name)

                with os.scandir(entry_path) as symlinks:
                    for link in symlinks:
                        os.unlink(link.path)

                os.rmdir(entry_path)

        os.rmdir(self.sequence_path)
        os.rmdir(self.path)

    def _load_calib(self, calibration):
        with os.scandir(os.path.join(calibration, "sequences")) as entries:
            for entry in entries:  # for each sequence
                entry_path = os.path.join(self.sequence_path, entry.name)
                os.mkdir(entry_path)

                with os.scandir(entry.path) as files:
                    for file in files:
                        source = os.path.abspath(file.path)
                        target = os.path.join(entry_path, file.name)
                        os.symlink(source, target)

    def _load_grayscale(self, grayscale):
        pass

    def _load_color(self, color):
        with os.scandir(os.path.join(color, "sequences")) as entries:
            for entry in entries:
                entry_path = os.path.join(self.sequence_path, entry.name)

                image_2_source = os.path.abspath(os.path.join(entry.path, "image_2"))
                image_2_target = os.path.join(entry_path, "image_2")
                os.symlink(image_2_source, image_2_target)

                image_3_source = os.path.abspath(os.path.join(entry.path, "image_3"))
                image_3_target = os.path.join(entry_path, "image_3")
                os.symlink(image_3_source, image_3_target)

    def _load_velodyne(self, velodyne):
        with os.scandir(os.path.join(velodyne, "sequences")) as entries:
            for entry in entries:
                entry_path = os.path.join(self.sequence_path, entry.name)

                source = os.path.abspath(os.path.join(entry.path, "velodyne"))
                target = os.path.join(entry_path, "velodyne")
                os.symlink(source, target)


if __name__ == '__main__':
    dataset = Dataset()
    calib = os.path.join(os.path.pardir, "dataset", "calibration")
    color = os.path.join(os.path.pardir, "dataset", "color")
    velodyne = os.path.join(os.path.pardir, "dataset", "velodyne")
    dataset.prepare_odometry(calib, color=color, velodyne=velodyne)

    odometry = dataset.get_odometry("00")
    print(odometry.sequence)
    print(odometry.get_cam2(2))

    dataset.close()
