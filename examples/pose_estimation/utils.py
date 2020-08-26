import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from configparser import ConfigParser
import json


# helper function to load camera configuration
def read_cameraConfig(path='/home/luo/workspace/thesis_ws/SoftRas/examples/pose_estimation/camera.ini'):
    """ 
    Read camera calibration matrix from path. 
    K = [ a_u   0   u_0
           0   a_v  v_0 
           0    0    1  ]

    @ return intrinsic matrix: np.array((3,3))
    """   
    parser = ConfigParser()
    parser.read(path)
    a_u, a_v, u_0, v_0, *frame_size =  json.loads(parser.get("camera", "config"))
    intrinsic_mat = np.array([[a_u,   0,   u_0 ],
                             [  0,   a_v,  v_0 ],
                             [  0,    0,    1  ]]) 
    # print(intrinsic_mat)
    # print(frame_size)
    return intrinsic_mat, frame_size


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
