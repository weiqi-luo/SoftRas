import math
import trimesh
from pyrender import IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer
import numpy as np
from numpy import linalg as LA 
from configparser import ConfigParser
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from PIL import Image
from os import path
from itertools import combinations_with_replacement
from mpl_toolkits.mplot3d import Axes3D
import sys
from tqdm import tqdm
import click
import os
# from tfaxis import TfAxis


def sample_sphere_random(r,n):
    print("Generating randomly {} points on a sphere of radius {} centered at the origin".format(n,r))
    xp=[];yp=[];zp=[]
    theta = np.random.uniform(0.0,2*np.pi,n)
    z = np.random.uniform(-r,r,n)
    for i in range (0,n):
        zp.append(z[i])
        xp.append(np.sqrt(r*r - z[i]*z[i])* np.cos(theta[i]))
        yp.append(np.sqrt(r*r - z[i]*z[i])* np.sin(theta[i]))
    tArr = np.transpose((xp,yp,zp))
    # drawing
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xp, yp, zp)
    plt.show()
    return (tArr,(xp,yp,zp))


def sample_sphere_grid(r,n):
    print("Generating fixed {} points on a sphere of radius {} centered at the origin".format(n,r))
    n_ = n
    xp=[];yp=[];zp=[]
    alpha = 4.0*np.pi*r*r/n
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    while True:
        for m in range (0,m_nu):
            nu = np.pi*(m+0.5)/m_nu
            m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
            for n in range (0,m_phi):
                phi = 2*np.pi*n/m_phi
                xp.append(r*np.sin(nu)*np.cos(phi))
                yp.append(r*np.sin(nu)*np.sin(phi))
                zp.append(r*np.cos(nu))
        tArr = np.transpose((xp,yp,zp))
        print(tArr.shape[0])
        if tArr.shape[0] >= n_:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xp, yp, zp)
            plt.show()
            return (tArr,(xp,yp,zp))
        else:
            n += (n - tArr.shape[0])
    

def transform_camera(t, n_rot):
    """Compute transform matrix for camera in order to pointing towards the object"""
    # compute euler angle
    H_list = []
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
    # rotate the images n_rot times
    for i in range(n_rot):
        rotAngle = i*2*math.pi/n_rot
        rotVec = t/LA.norm(t)*rotAngle
        R_rot = Rotation.from_rotvec(rotVec).as_dcm()
        H = np.vstack( (np.hstack( (np.dot(R_rot,R),t.reshape((3,1)))), np.array([0,0,0,1])))
        H_list.append(H)
    return H_list


def render_images(path_mesh, path_save, tArr, light, n_rot=None):
    """Render the images"""
    scene = Scene() # create scene
    object_trimesh = trimesh.load(path_mesh) # add object
    object_mesh = Mesh.from_trimesh(object_trimesh)
    scene.add(object_mesh, pose=np.identity(4))
    direc_l = DirectionalLight(color=np.ones(3), intensity=light) # create light
    spot_l = SpotLight(color=np.ones(3), intensity=light,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
    cam = IntrinsicsCamera(fx=570, fy=570, cx=64, cy=64) # create camera
    renderer = OffscreenRenderer(viewport_width=128, viewport_height=128) # create off-renderer
    count = 0
    pose_list = []
    n_samp = tArr.shape[0]
    if not n_rot:
        n_rot = round(36/2562*n_samp)
    print("Create dataset of {} points with {} times in-plane rotation".format(n_samp,n_rot))
    for t in tqdm(tArr): # render a photo for each pose of camera
        poses = transform_camera(t,n_rot) # compute the camera transformation matrix in order to pointing to object
        for pose in poses:
            light_node = scene.add(spot_l, pose=pose) # add light 
            cam_node = scene.add(cam, pose=pose) # add camera
            color, __ = renderer.render(scene) # render the image
            im = Image.fromarray(color) # save the image
            im.save(path.join(path_save,"{}.jpeg".format(count)))
            pose_list.append(pose)
            count += 1
            scene.remove_node(cam_node) # remove the camera and light node
            scene.remove_node(light_node)
    np.save(path.join(path_save,"camera_poses.npy"),pose_list) # save the camera poses


def test_sample(n,r):
    """Sample the sampling function and draw the sampling results"""
    # n = 800 # number of sampling points
    # r = 1 # radius
    # generate random points on the sphere
    __, (xp, yp, zp) = sample_sphere_grid(r,n) 
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(xp, yp, zp)
    # generate equally distributed points on the sphere 
    __, (xp, yp, zp) = sample_sphere_random(r,n) 
    ax = fig.add_subplot(212, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(xp, yp, zp)
    plt.show()

def test_cameraTransform():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_zlim3d([-1.5, 1.5])
    #
    r = 1
    n = 10
    n_rot = 4
    tArr, __ = sample_sphere_grid(r,n)
    for t in tArr:
        poses = transform_camera(t,n_rot)
        ax.plot3D((0,t[0]),(0,t[1]),(0,t[2]))
        for pose in poses:
            quat = Rotation.from_dcm(pose[0:3,0:3]).as_quat()
            tf = TfAxis(origin=t, quat=quat, scale=0.2)
            tf.plot(ax)
        plt.show()

def test_offrender():
    """Render the images with test view points (-1,-1,-1), (-1,-1,0) ..."""
    path_mesh = '../data/raw/models/019_pitcher_base/textured_simple.obj'
    path_save = "../data/interim"
    n_rot = 4
    tArr = list(combinations_with_replacement([1.5, 0, -1.5], 3)) # generate testing points (-1,-1,-1),(-1,-1,0)...
    tArr.remove((0,0,0)) # remove (0,0,0)
    light = 30
    tArr = np.asarray(tArr)
    render_images(path_mesh, path_save, tArr, light)

@click.command()
@click.argument('obj')
@click.argument('mode')
@click.argument('radius', default=1.5) #0.8
@click.argument('light', default=40)
@click.argument('num_sample', default=2562)

def release_offrender(obj, mode, radius, light, num_sample):
    """Render the images with automatically generated vieew points"""
    if mode == "train":
        tArr, __ = sample_sphere_grid(radius,num_sample) # sample n view points on the sphere of radius r 
        mode = ""
    elif mode == "test":
        tArr, __ = sample_sphere_random(radius,num_sample) # sample n view points on the sphere of radius r 
        mode = "_test"
    path_mesh = '../data/raw/models/{}/textured_simple.obj'.format(obj) # path of mesh
    path_save = "../data/{}{}".format(obj,mode)  # path to save rendered images
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    print("mode={} num_sample={} radius={} light={} path_save={}".format(mode, num_sample, radius, light, path_save))
    render_images(path_mesh, path_save, tArr, light) # generate synthetic photos
    

if __name__ == '__main__':
    # test_sample(2562,1.5)
    # test_cameraTransform()
    # test_offrender()
    # release_offrender("train")
    release_offrender()






