
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

import soft_renderer as sr


class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=True, fill_back=True, eps=1e-6,
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512, K=None,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(Renderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, K, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                        anti_aliasing, fill_back, eps)

    def forward(self, mesh, mode=None):
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)


class SoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512, K=None,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0], fov=None):
        super(SoftRenderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, K, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction,fov)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh, mode=None):
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)

    def render_fov(self, mesh, R, t, mode=None, background_color=[0,0,0]):
        mesh.reset_()

        # expand dimension 
        batch_size = R.shape[0]
        mesh.vertices = mesh.vertices.expand(batch_size, *mesh.vertices.shape[1:])
        mesh.faces = mesh.faces.expand(batch_size, *mesh.faces.shape[1:])
        mesh.textures = mesh.textures.expand(batch_size, *mesh.textures.shape[1:])

        self.set_texture_mode(mesh.texture_type)
        self.transform.set_transform(R=R,t=t)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        rgbd = self.rasterizer(mesh, mode)
        return rgbd[:,:3,:,:], rgbd[:,-1,:,:], rgbd[:,-1,:,:]

    def forward(self, vertices, faces, textures=None, mode=None, texture_type='surface', R=None, t=None):
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        return self.render_mesh(mesh, mode)