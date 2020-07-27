import math
import trimesh
from pyrender import IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer

import os
from os import path
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,IterableDataset
from torchvision import transforms, utils
from skimage.color import rgb2gray
import sys
import cv2
import imageio, random
from torchvision import transforms, utils

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
        self.transform = [Rescale(length), ToGrey(), Flatten(), ToTensor()]
        
    def __iter__(self):
        while True:
            theta = random.uniform(0.0,2*math.pi)
            z = random.uniform(-self.radius,self.radius)
            x = math.sqrt(self.radius**2 - z**2)* np.cos(theta)
            y = np.sqrt(self.radius**2 - z**2)* np.sin(theta)

            light_node = self.scene.add(self.spot_l, pose=pose) # add light 
            cam_node = self.scene.add(self.cam, pose=pose) # add camera
            color, depth = self.renderer.render(self.scene) # render the image
            self.scene.remove_node(cam_node) # remove the camera and light node
            self.scene.remove_node(light_node)

            image = self.transform(image)

            yield image
        

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

def show_cpose_batch(images_batch):
    """Show image with cpose for a batch of samples."""
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
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
    dataset = ObjectDataset(path_mesh="/home/luo/workspace/thesis_ws/SoftRas/data/YCB/006_mustard_bottle/textured_simple.obj", radius=1, length=28, light=100)
    dataloader = DataLoader(dataset, batch_size=5,shuffle=True, num_workers=4)
    plt.figure()
    for i_batch, sample_batched in enumerate(dataloader):
        # observe 4th batch and stop.
        show_cpose_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        if i_batch == 3:
            break    

    
if __name__ == '__main__':
    test_dataloader()
