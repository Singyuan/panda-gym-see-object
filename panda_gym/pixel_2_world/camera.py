import pybullet as p
import numpy as np
import math

class Camera(object):
    
    def __init__(self, image_size, near, far, fov):
        super().__init__()
        self.image_size = image_size
        self.near = near
        self.far = far
        self.fov = fov
        self.focal_length = (float(self.image_size[1])/2) / np.tan((np.pi * self.fov / 180) / 2)
        self.fov_height = (math.atan((float(self.image_size[0]) /2) / self.focal_length) * 2 / np.pi) * 180
        self.intrinsic_matrix, self.projection_matrix = self.compute_camera_matrix()
        
    def compute_camera_matrix(self):
        intrinsic_matrix = np.array(
            [[self.focal_length, 0, float(self.image_size[1]) / 2],
             [0, self.focal_length, float(self.image_size[0]) / 2],
             [0, 0, 1]]   
            )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov = self.fov_height,
            aspect = float(self.image_size[1]) / float(self.image_size[0]),
            nearVal = self.near,
            farVal = self.far
            )
        
        return intrinsic_matrix, projection_matrix
    
def cam_view2pose(cam_view_matrix):
    cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4,4).T)
    cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
    return cam_pose_matrix
    
def make_obs(camera, view_matrix):
    obs = p.getCameraImage(
        width = camera.image_size[1],
        height = camera.image_size[0],
        viewMatrix = view_matrix,
        projectionMatrix = camera.projection_matrix,
        renderer = p.ER_BULLET_HARDWARE_OPENGL,
        )
        
    need_convert = False
        
    if type(obs[2]) is tuple:
        need_convert = True
            
    if need_convert:
        rgb_pixels = np.asarray(obs[2]).reshape(camera.image_size[0], camera.image_size[1], 4)
        rgb_obs = rgb_pixels[:, :, :3]
        z_buffer = np.asarray(obs[3]).reshape(camera.image_size[0], camera.image_size[1])
        depth_obs = camera.far * camera.near / (camera.far - (camera.far - camera.near) * z_buffer)
        mask_obs = np.asarray(obs[4]).reshape(camera.image_size[0], camera.image_size[1])
    else:
        rgb_obs = obs[2][:, :, :3]
        depth_obs = camera.far * camera.near / (camera.far - (camera.far - camera.near) * obs[3])
        mask_obs = obs[4]
            
    mask_obs[mask_obs == -1] = 0
    return rgb_obs.astype(np.uint8), depth_obs, mask_obs