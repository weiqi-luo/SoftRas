import math
import trimesh
from pyrender import IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer
from numpy import linalg as LA 
from scipy.spatial.transform import Rotation

import soft_renderer as sr
import examples as ex
import soft_renderer.functional as srf

import os
from os import path
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,IterableDataset
from torchvision import transforms
import torchvision
from skimage.color import rgb2gray
import sys
import cv2
import imageio, random
from utils import read_cameraConfig

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ObjectDataset(IterableDataset):
    """Face cpose dataset."""

    def __init__(self, path_mesh, radius, length, light):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.radius = radius
        self.scene = Scene() # create scene
        object_trimesh = trimesh.load(path_mesh) # add object
        object_mesh = Mesh.from_trimesh(object_trimesh)
        self.scene.add(object_mesh, pose=np.identity(4))
        self.direc_l = DirectionalLight(color=np.ones(3), intensity=light) # create light
        self.spot_l = SpotLight(color=np.ones(3), intensity=light,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
        self.cam = IntrinsicsCamera(fx=570, fy=570, cx=64, cy=64) # create camera
        self.renderer = OffscreenRenderer(viewport_width=128, viewport_height=128) # create off-renderer
        self.transform = transforms.Compose([Rescale(length), ToTensor()])
        
    def __iter__(self):
        while True:
            theta = random.uniform(0.0,2*math.pi)
            z = random.uniform(-self.radius,self.radius)
            x = math.sqrt(self.radius**2 - z**2)* np.cos(theta)
            y = np.sqrt(self.radius**2 - z**2)* np.sin(theta)
            t = np.array((x,y,z))

            axisZ = np.array([0,0,1]) # z axis of camera coordinate
            rotAngle = math.acos(np.dot(axisZ,t)/(LA.norm(axisZ)*LA.norm(t))) # compute the rotation angle in order to align axisZ with t 
            if rotAngle == 0: # when axisZ is already aligned with t, do not rotate 
                R = np.identity(3)
            elif rotAngle == math.pi: # when axisZ is inverse to t, rotate pi on y axis
                R = np.array([[math.cos(math.pi), 0, math.sin(math.pi)],
                    [0, 1, 0],
                    [-math.sin(math.pi), 0, math.cos(math.pi)]])
            else: # otherwise compute angle-axis vector and convert it to rotation matrix
                rotVec = np.cross(axisZ,t)
                rotVec = rotVec/LA.norm(rotVec)*rotAngle
                R = Rotation.from_rotvec(rotVec).as_dcm()
            
                rotAngle = random.uniform(0.0,2*math.pi)
                rotVec = t/LA.norm(t)*rotAngle
                R_rot = Rotation.from_rotvec(rotVec).as_dcm()
                pose = np.vstack( (np.hstack( (np.dot(R_rot,R),t.reshape((3,1)))), np.array([0,0,0,1])))

            light_node = self.scene.add(self.spot_l, pose=pose) # add light 
            cam_node = self.scene.add(self.cam, pose=pose) # add camera
            color, depth = self.renderer.render(self.scene) # render the image
            self.scene.remove_node(cam_node) # remove the camera and light node
            self.scene.remove_node(light_node)
            
            color = self.transform(color)

            yield color,pose
        

class Rescale(object):
    """Rescale the image in a sample to a given size."""
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        image = transform.resize(image, (self.output_size, self.output_size))
        return image


class RandomCrop(object):
    """Crop randomly the image in a sample."""
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # print("top is {0}, left is {1}".format(top, left))
        image = image[top: top + new_h,
                      left: left + new_w]
        return image


class Nomalize(object):
    def __call__(self, image):                
        image = image / 255   
        return image


class ToGrey(object):
    def __call__(self, image):        
        image = rgb2gray(image)
        return image


class ToRGB(object):
    def __call__(self, image):        
        image = np.stack([image,image,image],axis=2)
        return image


class Flatten():
    def __call__(self, image):        
        image = np.reshape(image,(-1,image.shape[0]*image.shape[1]))
        return image


class ExpandDim(object):
    def __call__(self, image):        
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):       
        if image.ndim == 3:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
            image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()


# ============================ TEST ============================ #

def show_batch(images_batch):
    """Show image with cpose for a batch of samples."""
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = torchvision.utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


def test_objectPoseDataset():
    object_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                        root_dir='../data/interim/')
    fig = plt.figure()
    for i,sample in enumerate(object_dataset):
        print(i, sample['image'].shape, sample['cpose'].shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])
        if i == 3:
            break
    plt.show()


def test_transform():
    object_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                        root_dir='../data/interim/')
    scale = Rescale(256)
    crop = RandomCrop(100)
    composed = transforms.Compose([Rescale(256),
                                RandomCrop(224)])
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = object_dataset[11]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        plt.imshow(transformed_sample['image'])
        print(transformed_sample['image'].shape)
    plt.show()


def test_transformDataset():
    transformed_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                           root_dir='../data/interim/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()]))
    for i,sample in enumerate(transformed_dataset):
        print(i, sample['image'].size(), sample['cpose'].size())
        if i == 3:
            break
    

def test_dataloader():
    dataset = ObjectDataset(path_mesh="/home/luo/workspace/thesis_ws/SoftRas/data/YCB/019_pitcher_base/textured_simple.obj", radius=2, length=128, light=100)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    plt.figure()
    for i_batch, sample_batched in enumerate(dataloader):
    #     # observe 4th batch and stop.
        show_cpose_batch(sample_batched)
        plt.show()
        if i_batch == 3:
            break    

def test_renderer():
    path_mesh = "/home/luo/workspace/thesis_ws/SoftRas/data/YCB/019_pitcher_base/textured_simple.obj"
    dataset = ObjectDataset(path_mesh=path_mesh, radius=2, length=128, light=100)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # load from Wavefront .obj file
    mesh = sr.Mesh.from_obj(path_mesh, load_texture=True, texture_res=5, texture_type='surface')
    
    # create renderer with SoftRas
    # pose = np.eye(3,4)
    # renderer = sr.SoftRenderer(image_size=128, sigma_val=1e-4, aggr_func_rgb='hard', 
    #                            camera_mode='projection', viewing_angle=15, P=pose)

    # other settings
    camera_distance = 2
    elevation = 30
    azimuth = 0   

    image_size=256; background_color=[0,0,0]; near=1; far=100; 
    anti_aliasing=False; fill_back=True; eps=1e-3
    sigma_val=1e-5; dist_func='euclidean'; dist_eps=1e-4
    gamma_val=1e-4; aggr_func_rgb='softmax'; aggr_func_alpha='prod'
    texture_type='surface'
    
    camera_mode='projection'
    dist_coeffs=None; orig_size=512
    perspective=True; viewing_angle=30; viewing_scale=1.0
    eye=None; camera_direction=[0,0,1]

    light_mode='surface'
    light_intensity_ambient=0.5; light_color_ambient=[1,1,1]
    light_intensity_directionals=0.5; light_color_directionals=[1,1,1]
    light_directions=[0,1,0] ; mode=None
    # P = np.eye(3,4)
    # P = np.expand_dims(P,axis=0)
    K, frame_size = read_cameraConfig()


    dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(1, 1)
    orig_size = 512
    # light
    lighting = sr.Lighting(light_mode,
                                light_intensity_ambient, light_color_ambient,
                                light_intensity_directionals, light_color_directionals,
                                light_directions)

    # # camera
    # transform = sr.Transform(camera_mode, 
    #                                 P, dist_coeffs, orig_size,
    #                                 perspective, viewing_angle, viewing_scale, 
    #                                 eye, camera_direction)

    # rasterization
    rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                        anti_aliasing, fill_back, eps,
                                        sigma_val, dist_func, dist_eps,
                                        gamma_val, aggr_func_rgb, aggr_func_alpha,
                                        texture_type)

    lighting.light_mode = texture_type
    rasterizer.texture_type = texture_type



    for (color,pose) in dataloader:
        plt.figure(0)
        show_batch(color)
        
        # mesh = lighting(mesh)
        
        # # mesh.reset_()
        # for R in getRotation(10):
        #     pose = np.matmul(pose, R)
        #     pose = pose[:,:3,:]
        #     P = pose
        #     P = np.matmul(K,pose)
            
        #     # pose = pose.double()
        #     # renderer.transform.set_eyes(pose) 
        #     # renderer = sr.SoftRenderer(image_size=128, sigma_val=1e-4, aggr_func_rgb='hard', 
        #     #                         camera_mode='projection', viewing_angle=15, P=pose)
        #     # images = renderer.render_mesh(mesh)
        #     # >

        #     # >
        #     # mesh = transform(mesh)
        #     # mesh.vertices = srf.projection(mesh.vertices, P, dist_coeffs, orig_size)
        #     vertices = mesh.vertices
        #     vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
        #     P = P.transpose(2,1)
        #     P = P.cuda().float()
        #     vertices = torch.bmm(vertices, P)

        #     x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        #     x_ = x / (z + 1e-5)
        #     y_ = y / (z + 1e-5)

        #     # Get distortion coefficients from vector
        #     k1 = dist_coeffs[:, None, 0]
        #     k2 = dist_coeffs[:, None, 1]
        #     p1 = dist_coeffs[:, None, 2]
        #     p2 = dist_coeffs[:, None, 3]
        #     k3 = dist_coeffs[:, None, 4]

        #     # we use x_ for x' and x__ for x'' etc.
        #     r = torch.sqrt(x_ ** 2 + y_ ** 2)
        #     x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
        #     y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
        #     x__ = 2 * (x__ - frame_size[1] / 2.) / frame_size[1]
        #     y__ = 2 * (y__ - frame_size[0] / 2.) / frame_size[0]

        #     vertices = torch.stack([x__,y__,z], dim=-1)
        #     mesh.vertices = vertices

        # > 
            # instead of P*x we compute x'*P'
        vertices = torch.matmul(vertices, R.transpose(2,1)) + t
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        # Get distortion coefficients from vector
        k1 = dist_coeffs[:, None, 0]
        k2 = dist_coeffs[:, None, 1]
        p1 = dist_coeffs[:, None, 2]
        p2 = dist_coeffs[:, None, 3]
        k3 = dist_coeffs[:, None, 4]

        # we use x_ for x' and x__ for x'' etc.
        r = torch.sqrt(x_ ** 2 + y_ ** 2)
        x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
        y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
        vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
        vertices = torch.matmul(vertices, K.transpose(1,2))
        u, v = vertices[:, :, 0], vertices[:, :, 1]
        v = orig_size - v
        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        u = 2 * (u - orig_size / 2.) / orig_size
        v = 2 * (v - orig_size / 2.) / orig_size
        vertices = torch.stack([u, v, z], dim=-1)
            
        # >
        images = rasterizer(mesh, mode)

        # >
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        image = (255*image).astype(np.uint8)
        plt.figure(1)
        if np.sum(image):
            plt.imshow(image)
            plt.show()
    
def getRotation(n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = math.pi*2/n*i   
                y = math.pi*2/n*j   
                z = math.pi*2/n*k   
                Rx = np.array([[1, 0,           0,            0],
                               [0, math.cos(x), -math.sin(x), 0],
                               [0, math.sin(x),  math.cos(x), 0],
                               [0, 0,           0,            1]  ])
                Ry = np.array([[ math.cos(y), 0, math.sin(y), 0],
                               [0,            1, 0,           0],
                               [-math.sin(y), 0, math.cos(y), 0],
                               [0,            0, 0,           1]  ])
                Rz = np.array([[math.cos(z), -math.sin(z), 0, 0],
                               [math.sin(z),  math.cos(z), 0, 0],
                               [0,           0,            1, 0],
                               [0,           0,            0, 1]  ])
                print(x,y,z)
                yield Rz @ Ry @ Rx

if __name__ == '__main__':
    test_renderer()
