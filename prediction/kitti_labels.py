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
    0 :         [0, 0, 0],                  #   unlabeled
    1 :         [0, 0, 255],                #   outlier
    10:         [245, 150, 100],            #   car
    11:         [245, 230, 100],            #   bicycle
    13:         [250, 80, 100],             #   bus
    15:         [150, 60, 30],              #   motorcycle
    16:         [255, 0, 0],                #   on-rails
    18:         [180, 30, 80],              #   truck
    20:         [255, 0, 0],                #   other-vehicle
    30:         [30, 30, 255],              #   person
    31:         [200, 40, 255],             #   bicyclist
    32:         [90, 30, 150],              #   motorcyclist
    40:         [255, 0, 255],              #   road
    44:         [255, 150, 255],            #   parking
    48:         [75, 0, 75],                #   sidewalk
    49:         [75, 0, 175],               #   other-ground
    50:         [0, 200, 255],              #   building
    51:         [50, 120, 255],             #   fence
    52:         [0, 150, 255],              #   other-structure
    60:         [170, 255, 150],            #   lane-marking
    70:         [0, 175, 0],                #   vegetation
    71:         [0, 60, 135],               #   trunk
    72:         [80, 240, 150],             #   terrain
    80:         [150, 240, 255],            #   pole
    81:         [0, 0, 255],                #   traffic-sign
    99:         [255, 255, 50],             #   other-object
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
    12          :           31      ,       #   rider           -> bicyclist
    13          :           10      ,       #   car
    14          :           18      ,       #   truck
    15          :           13      ,       #   bus
    16          :           20      ,       #   train           -> other vehicle
    17          :           15      ,       #   motorcycle
    18          :           11      ,       #   bicycle
    255         :           0       ,       #   unlabeled
}