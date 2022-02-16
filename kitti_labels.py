import numpy as np


kitti_labels = {
    #   id          name
    0:          "unlabeled",
    1:          "outlier",
    10:         "car",
    11:         "bicycle",
    13:         "bus",
    15:         "motorcycle",
    16:         "on-rails",
    18:         "truck",
    20:         "other-vehicle",
    30:         "person",
    31:         "bicyclist",
    32:         "motorcyclist",
    40:         "road",
    44:         "parking",
    48:         "sidewalk",
    49:         "other-ground",
    50:         "building",
    51:         "fence",
    52:         "other-structure",
    60:         "lane-marking",
    70:         "vegetation",
    71:         "trunk",
    72:         "terrain",
    80:         "pole",
    81:         "traffic-sign",
    99:         "other-object"
}

kitti_colors = {
    #   id          color
    0:         [0, 0, 0],  # unlabeled
    1:         [0, 0, 255],  # outlier
    10:        [245, 150, 100],  # car
    11:        [245, 230, 100],  # bicycle
    13:        [250, 80, 100],  # bus
    15:        [150, 60, 30],  # motorcycle
    16:        [255, 0, 0],  # on-rails
    18:        [180, 30, 80],  # truck
    20:        [255, 0, 0],  # other-vehicle
    30:        [30, 30, 255],  # person
    31:        [200, 40, 255],  # bicyclist
    32:        [90, 30, 150],  # motorcyclist
    40:        [255, 0, 255],  # road
    44:        [255, 150, 255],  # parking
    48:        [75, 0, 75],  # sidewalk
    49:        [75, 0, 175],  # other-ground
    50:        [0, 200, 255],  # building
    51:        [50, 120, 255],  # fence
    52:        [0, 150, 255],  # other-structure
    60:        [170, 255, 150],  # lane-marking
    70:        [0, 175, 0],  # vegetation
    71:        [0, 60, 135],  # trunk
    72:        [80, 240, 150],  # terrain
    80:        [150, 240, 255],  # pole
    81:        [0, 0, 255],  # traffic-sign
    99:        [255, 255, 50],  # other-object
}

cityscapes2kitti = {
    # cityscapes trainId      kitti id            name
    0:           40,  # road
    1:           48,  # sidewalk
    2:           50,  # building
    3:           52,  # wall            ->  other structure
    4:           51,  # fence
    5:           80,  # pole
    6:           81,  # traffic light   ->  traffic sign
    7:           81,  # traffic sign
    8:           70,  # vegetation
    9:           72,  # terrain
    10:          1,  # sky             ->  outlier
    11:          30,  # person
    12:          31,  # rider           -> bicyclist
    13:          10,  # car
    14:          18,  # truck
    15:          13,  # bus
    16:          20,  # train           -> other vehicle
    17:          15,  # motorcycle
    18:          11,  # bicycle
    255:         0,  # unlabeled
}

kitti_label2learning = {
    0: 0,      # "unlabeled"
    1: 0,      # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

kitti_learning2index = {
    0: 0,      # "unlabeled", and others ignored
    1: 10,     # "car"
    2: 11,     # "bicycle"
    3: 15,     # "motorcycle"
    4: 18,     # "truck"
    5: 20,     # "other-vehicle"
    6: 30,     # "person"
    7: 31,     # "bicyclist"
    8: 32,     # "motorcyclist"
    9: 40,     # "road"
    10: 44,    # "parking"
    11: 48,    # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81     # "traffic-sign"
}


def city2kitti_translate(label):
    # Terminate at the last one dimension
    if len(label.shape) == 1:
        return np.array([cityscapes2kitti[l] for l in label])

    return np.array([city2kitti_translate(l) for l in label])


def kitti_colorize(label):
    # Terminate at the last one dimension
    if len(label.shape) == 1:
        return np.array([kitti_colors[l] for l in label])

    return np.array([kitti_colorize(l) for l in label])
