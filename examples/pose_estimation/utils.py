from preprocess import *
import torch
from torch.utils.data import DataLoader
import os.path as path
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import numpy as np
import sys
import math
import copy
from models import *
from scipy.spatial.transform import Rotation
import cv2
from skimage import io
import time
import trimesh
from pyrender import IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer
from numpy import linalg as LA 


# helper functions to show an image
def imshow(img, a):
    a.clear()
    npimg = img.detach().to('cpu').numpy()
    if npimg.ndim==3 and npimg.shape[0]==3:
        npimg = np.transpose(npimg, (1, 2, 0))
        a.imshow(npimg); a.set_xticks(()); a.set_yticks(())
    elif npimg.ndim==3 and npimg.shape[0]==1 or npimg.ndim==1: 
        a.imshow(npimg.reshape((npimg.shape[1],npimg.shape[2])),cmap='gray'); a.set_xticks(()); a.set_yticks(())
    elif npimg.ndim==2:
        length = int(math.sqrt(npimg.shape[1]))
        a.imshow(npimg.reshape((length,length)),cmap='gray'); a.set_xticks(()); a.set_yticks(())
    # print(npimg.shape, np.min(npimg),np.max(npimg))
    

def get_transform(model,dataset,length,augment,test=None):
    if model in ["convrgb"+str(i) for i in range(3)]:
        if dataset == "mnist":
            tran = [Rescale(length), ToRGB(), ToTensor()]
        else:    
            tran = [Rescale(length), ToTensor()]
    elif model == "conv": 
        if dataset == "mnist":
            tran = [Rescale(length), ExpandDim(), ToTensor()]
        else:
            tran = [Rescale(length), ToGrey(), ExpandDim(), ToTensor()]
    elif model == "linear":
        tran = [Rescale(length), ToGrey(), Flatten(), ToTensor()]
    else:
        print("Entered wrong model")
        sys.exit()
    tran_aug = [  ## TODO
        
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=(0,0), translate=None, scale=(0.5,1), shear=None, resample=False, fillcolor=(255,255,255)),
        transforms.RandomResizedCrop(size = length, ratio=(1.0,1.0), scale=(0.3,1.0)),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor()]
    if not augment: 
        tran_aug = tran
    if test: 
        return (transforms.Compose(tran[:-1]), transforms.Compose(tran_aug[:-1])) 
    return (transforms.Compose(tran), transforms.Compose(tran_aug))
    


def get_model(MODEL, LENGTH):
    if MODEL == "convrgb2":
        print("Convolutional rgb version")
        assert not LENGTH%16
        return AutoEncoder_convrgb2(LENGTH)
    if MODEL == "convrgb1":
        print("Convolutional rgb version")
        assert not LENGTH%4
        return AutoEncoder_convrgb1(LENGTH)
    if MODEL == "convrgb0":
        print("Convolutional rgb version")
        assert not LENGTH%4
        return AutoEncoder_convrgb0(LENGTH)
    elif MODEL == "conv": 
        print("Convolusional grey version")
        return AutoEncoder_conv(LENGTH)
    elif MODEL == "linear":
        print("Linear version")
        return AutoEncoder_linear(LENGTH)


def load_model_dataset(filename,suffix,BATCH_SIZE,augment=None):
    checkpoint = torch.load(filename)
    # Load and normalize the tesing dataset
    if augment:
        transform_aug = checkpoint["transform_aug"]
    else:
        transform_aug = checkpoint["transform"]
    if suffix:
        root_dir =  path.join("../data/", "{}_{}".format(checkpoint['info']['dataset'],suffix))
    else:
        root_dir =  path.join("../data/", checkpoint['info']['dataset'])
    print("Loading dataset from ",root_dir)
    npy_file =  path.join(root_dir, "camera_poses.npy")
    dataset = ObjectPoseDataset(npy_file=npy_file, root_dir=root_dir, 
        transform=checkpoint['transform'], transform_aug=augment)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # Load Convolutional Neural Network
    autoencoder = checkpoint['autoencoder']
    autoencoder.load_state_dict(checkpoint['state_dict'])
    print(autoencoder)
    autoencoder.eval()
    info = checkpoint['info']
    info['root_dir'] = root_dir
    info['transform'] = checkpoint['transform']
    return autoencoder,dataloader,info


def cos_sim(x,y):
    res = np.inner(x,y)/(LA.norm(x) * LA.norm(y))
    return res


def rotation_error(R1, R2):
    R1 = R1.reshape((4,4))[:3,:3]
    R2 = R2.reshape((4,4))[:3,:3]
    R_error = np.matmul(R1, np.transpose(R2))
    R_error = Rotation.from_matrix(R_error)
    R_error = R_error.as_rotvec()
    return LA.norm(R_error)


class visualize_poseEstimation():
    def __init__(self,path_mesh,light=40):
        self.scene = Scene() # create scene
        object_trimesh = trimesh.load(path_mesh) # add object
        object_mesh = Mesh.from_trimesh(object_trimesh)
        self.scene.add(object_mesh, pose=np.identity(4))
        self.direc_l = DirectionalLight(color=np.ones(3), intensity=light) # create light
        self.spot_l = SpotLight(color=np.ones(3), intensity=light,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
        self.cam = IntrinsicsCamera(fx=570, fy=570, cx=64, cy=64) # create camera
        self.renderer = OffscreenRenderer(viewport_width=128, viewport_height=128) # create off-renderer

    def call(self,pose):
        cam_node = self.scene.add(self.cam, pose=pose)
        light_node = self.scene.add(self.spot_l, pose=pose) # add light         
        color, _ = self.renderer.render(self.scene)
        self.scene.remove_node(cam_node)
        self.scene.remove_node(light_node)
        return color


def test_augment():
    img_path = "../data/024_bowl/0.jpeg"
    img_ori = cv2.imread(img_path)
    key = cv2.waitKey(10)
    cv2.imshow("ori",img_ori)
    while True:
        transform, transform_aug = get_transform("convrgb2","bowl",32,True,True)
        img_resize = np.array(transform(img_ori))
        img_aug = np.array(transform_aug(img_ori))
        cv2.imshow("augment",img_aug)  
        cv2.imshow("resized",img_resize)
        key = cv2.waitKey(10)
        time.sleep(1)
        if key==27:
            return


if __name__ == "__main__":
    test_augment()