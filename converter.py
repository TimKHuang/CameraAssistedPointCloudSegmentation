import numpy as np

class Converter:
    """
    Class to convert points between different coordinate systems

    Attributes
    ----------
    calib : 

    Methods
    -------
    set_fov :
        set the vertical and horizontal field of view
    lidar_to_image :
        Convert lidar points to image pixels of the same frame
    lidar_to_lidar:
        Convert lidar points from one frame to another
    """

    def __init__(self, calib):
        self.tr = calib.T_cam0_velo
        self.velo2cam = calib.T_cam2_velo
        self.cam2img = calib.K_cam2
        self.fov_initialized = False
    

    def set_fov(self, v_fov=(-24.9, 2.0), h_fov=(-45, 45)):
        self.v_fov = v_fov
        self.h_fov = h_fov
        self.fov_initialized = True


    def in_view_point(self, points):
        
        if not self.fov_initialized:
            raise Exception("Functional call before setting fields of view.")
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        h_in_range = np.logical_and(np.arctan2(y, x) < (self.h_fov[1] * np.pi / 180),
                                    np.arctan2(y, x) > (self.h_fov[0] * np.pi / 180))
        v_in_range = np.logical_and(np.arctan2(z, x) < (self.v_fov[1] * np.pi / 180),
                                    np.arctan2(z, x) > (self.v_fov[0] * np.pi / 180))
        mask = np.logical_and(h_in_range, v_in_range)

        return mask

    
    def lidar_to_image(self, points):
        
        # Convert to Camera coordinate
        cam = self.velo2cam @ points.T
        cam = np.delete(cam, 3, axis=0)

        # Convert to Image coordinate
        img = self.cam2img @ cam
        img = img[::] / img[::][2]
        img = np.delete(img, 2, axis=0)

        return img.T


    def lidar_to_lidar(self, source_points, source_pose, target_pose):
        
        # save the remission
        remissions = source_points[:, 3]
        points = np.ones(source_points.shape)
        points[:, 0:3] = source_points[:, 0:3]

        # transformation matrix
        tr_inv = np.linalg.inv(self.tr)
        source = tr_inv @ source_pose @ self.tr
        target = tr_inv @ target_pose @ self.tr
        diff = np.linalg.inv(target) @ source

        target_points = (diff @ points.T).T
        target_points[:, 3] = remissions

        return target_points

