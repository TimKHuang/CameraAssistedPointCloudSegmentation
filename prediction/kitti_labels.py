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

cityscapes2kitti = {
#   cityscapes trainId      kitti id            name
    0           :           40      ,       #   road
    1           :           48      ,       #   sidewalk
    2           :           50      ,       #   building
    3           :           52      ,       #   wall            ->  other structure
    4           :           51      ,       #   fence
    5           :           80      ,       #   pole
    6           :           81      ,       #   traffic light   ->  traffic sign
    7           :           81      ,       #   traffic sign
    8           :           70      ,       #   vegetation
    9           :           72      ,       #   terrain
    10          :           1       ,       #   sky             ->  outlier
    11          :           30      ,       #   person
    12          :           31      ,       #   rider           -> bicyclist ( TODO what about motorcylist)
    13          :           10      ,       #   car
    14          :           18      ,       #   truck
    15          :           13      ,       #   bus
    16          :           20      ,       #   train           -> other vehicle
    17          :           15      ,       #   motorcycle
    18          :           11      ,       #   bicycle
    255         :           0       ,       #   unlabeled
}