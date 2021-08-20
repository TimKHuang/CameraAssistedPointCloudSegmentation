import cv2
import numpy as np


def distance_color(value, minimum=0, maximum=120):
    value = np.clip(value, minimum, maximum)
    return ((value - minimum) / (maximum - minimum) * 120).astype(np.uint8)


def plot_point_on_image(image, xyz_origin, xy_projected):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    x = xyz_origin[0, :]
    y = xyz_origin[1, :]
    z = xyz_origin[2, :]
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = distance_color(distance)

    for i in range(xy_projected.shape[1]):
        coord_x = np.int32(xy_projected[0][i])
        coord_y = np.int32(xy_projected[1][i])
        cv2.circle(hsv_image, (coord_x, coord_y), 1, (int(color[i]), 255, 255), -1)

    result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return result
