
import math
import numpy as np
import torch
import torch.nn as nn

import soft_renderer.functional as srf

class Projection(nn.Module):
    def __init__(self, P, dist_coeffs=None, orig_size=512):
        super(Projection, self).__init__()

        self.P = P
        self.dist_coeffs = dist_coeffs
        self.orig_size = orig_size

        if isinstance(self.P, np.ndarray):
            self.P = torch.from_numpy(self.P).cuda()
        if self.P is None or self.P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
            raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
        if dist_coeffs is None:
            self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(self.P.shape[0], 1)

    def forward(self, vertices):
        vertices = srf.projection(vertices, self.P, self.dist_coeffs, self.orig_size)
        return vertices

class Projection_fov(nn.Module):
    def __init__(self, fov):
        super(Projection_fov, self).__init__()
        self.width = math.tan(fov/2)

    def forward(self, vertices, eps=1e-5):
        # camera transform
        # vertices = torch.matmul(vertices, R.transpose(2,1)) + t
        if isinstance(self.t, np.ndarray):
            self.t = torch.cuda.FloatTensor(self.t)
        if isinstance(self.R, np.ndarray):
            self.R = torch.cuda.FloatTensor(self.R)
        self.t = self.t.view(-1,1,3)
        vertices = vertices - self.t
        vertices = torch.matmul(vertices, self.R.transpose(1,2))
        
        # compute fov
        # compute perspective distortion
        z = vertices[:, :, 2]
        x = vertices[:, :, 0] / z / self.width
        y = vertices[:, :, 1] / z / self.width
        vertices = torch.stack((x,y,z), dim=2)
        return vertices

class LookAt(nn.Module):
    def __init__(self, perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(LookAt, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        vertices = srf.look_at(vertices, self._eye)
        # perspective transformation
        if self.perspective:
            vertices = srf.perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = srf.orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Look(nn.Module):
    def __init__(self, camera_direction=[0,0,1], perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(Look, self).__init__()
        
        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye
        self.camera_direction = camera_direction

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        vertices = srf.look(vertices, self._eye, self.camera_direction)
        # perspective transformation
        if self.perspective:
            vertices = srf.perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = srf.orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Transform(nn.Module):
    def __init__(self, camera_mode='projection', P=None, K=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],fov=None):
        super(Transform, self).__init__()

        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.transformer = Projection(P, dist_coeffs, orig_size)
        if self.camera_mode == 'projection_fov':
            self.transformer = Projection_fov(fov)
        elif self.camera_mode == 'look':
            self.transformer = Look(perspective, viewing_angle, viewing_scale, eye, camera_direction)
        elif self.camera_mode == 'look_at':
            self.transformer = LookAt(perspective, viewing_angle, viewing_scale, eye)
        else:
            raise ValueError('Camera mode has to be one of projection, projection_fov, look or look_at')

    def forward(self, mesh):
        mesh.vertices = self.transformer(mesh.vertices)
        return mesh

    def set_eyes_from_angles(self, distances, elevations, azimuths):
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = srf.get_points_from_angles(distances, elevations, azimuths)

    def set_eyes(self, eyes):
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = eyes

    def set_transform(self, R, t):
        if self.camera_mode != "projection_fov":
            raise ValueError('Projection does not need to set eyes')
        self.transformer.R = R
        self.transformer.t = t

    @property
    def eyes(self):
        return self.transformer._eyes
    
