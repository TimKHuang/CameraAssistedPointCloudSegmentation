import pykitti
import os


class Dataset:

    def __init__(self, store_path = None):
        if store_path is None:
            store_path = os.path.curdir
        self.path = os.path.abspath(os.path.join(store_path, "temp"))
        os.mkdir(self.path)

        self.sequence_path = os.path.join(self.path, "sequences")
        os.mkdir(self.sequence_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def open_odometry(calibration, store_path=None, grayscale=None, color=None, velodyne=None):
        dataset = Dataset(store_path=store_path)
        dataset.prepare_odometry(calibration, grayscale=grayscale, color=color, velodyne=velodyne)
        return dataset

    def prepare_odometry(self, calibration, grayscale=None, color=None, velodyne=None):
        self._load_calib(calibration)

        if grayscale is not None:
            self._load_grayscale(grayscale)

        if color is not None:
            self._load_color(color)

        if velodyne is not None:
            self._load_velodyne(velodyne)

    def get_sequence(self, sequence):
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
        print("Temporary Folder Deleted")

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
    calib = os.path.join(os.path.pardir, os.path.pardir, "dataset", "calibration")
    color = os.path.join(os.path.pardir, os.path.pardir, "dataset", "color")
    velodyne = os.path.join(os.path.pardir, os.path.pardir, "dataset", "velodyne")

    with Dataset.open_odometry(calib, color=color, velodyne=velodyne) as dataset:
        sequence = dataset.get_sequence("00")

        print(sequence.get_velo(0).shape)
        print(sequence.calib.T_cam2_velo)
