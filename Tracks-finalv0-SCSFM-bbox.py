#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pdb
from collections import defaultdict 
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter  
import matplotlib.patches as patches


import time

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import argoverse.visualization.visualization_utils as viz_util


# In[ ]:


import torch

from imageio import imread, imsave
from skimage.transform import resize as imresize
#from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispResNet
import models
from utils import tensor2array

from matplotlib import pyplot as plt
import pdb

from inverse_warp import pose_vec2mat
import pandas as pd


# In[ ]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[ ]:


RESNET_LAYERS = 18

if RESNET_LAYERS == 18:
    DISPNET_PRETRAINED = './Pretrained_models/SC-SFM-L_/resnet18_depth_256/09-05-23:14/dispnet_model_best.pth.tar'
    POSENET_PRETRAINED = './Pretrained_models/SC-SFM-L_/resnet18_depth_256/09-05-23:14/exp_pose_model_best.pth.tar'
if RESNET_LAYERS == 50:
    DISPNET_PRETRAINED = '/Users/raghavamodhugu/Desktop/IIITH/Trajectory_prediction/argoverse_tracks_generation/SC-SfMLearner-Release/Pretrained_models/SC-SFM-L_/resnet50_depth_256_argo/09-06-13:59/dispnet_model_best.pth.tar'
    POSENET_PRETRAINED = '/Users/raghavamodhugu/Desktop/IIITH/Trajectory_prediction/argoverse_tracks_generation/SC-SfMLearner-Release/Pretrained_models/SC-SFM-L_/resnet50_depth_256_argo/09-06-13:59/exp_pose_model_best.pth.tar'

ORIGINAL_HEIGHT = 1024
ORIGINAL_WIDTH = 1792

RESIZE_H = 256
RESIZE_W = 448


# In[ ]:


disp_net = DispResNet(RESNET_LAYERS, False).to(device)
weights = torch.load(DISPNET_PRETRAINED , map_location=device)
disp_net.load_state_dict(weights['state_dict'])
disp_net.eval()

pose_net = models.PoseResNet(RESNET_LAYERS, False).to(device)
weights = torch.load(POSENET_PRETRAINED , map_location=device)
pose_net.load_state_dict(weights['state_dict'], strict=False)
pose_net.eval()


# In[ ]:


def get_resized_img(file, rh=256, rw=448, h=1024, w=1792):
    img = imread(file).astype(np.float32)[:h,:w]
    img = imresize(img, (rh, rw), anti_aliasing=True).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = torch.from_numpy(img).unsqueeze(0)
    return tensor_img


# In[ ]:


[file.split('_')[0] for file in os.listdir() if file[-4:] == '.csv']


# In[ ]:


root_dir =  '/Users/raghavamodhugu/Downloads/Argoverse_samples/argoverse-tracking/sample'
argoverse_loader = ArgoverseTrackingLoader(root_dir)
camera = argoverse_loader.CAMERA_LIST[0]


# In[ ]:


def custom_transform_point_cloud(point_cloud, frame, log):
    """Apply the SE(3) transformation to this point cloud.
    Args:
        point_cloud: Array of shape (N, 3)
    Returns:
        transformed_point_cloud: Array of shape (N, 3)
    """
    transform_matrix = pd.read_csv('{}_CameraTrajectory.csv'.format(log), header=None).values[frame].reshape(3,4) 
    transform_matrix = np.vstack((transform_matrix, np.array([0.0, 0.0, 0.0, 1.0])))
    
    # convert to homogeneous
    num_pts = point_cloud.shape[0]
    homogeneous_pts = np.hstack([point_cloud, np.ones((num_pts, 1))])
    transformed_point_cloud = homogeneous_pts.dot(transform_matrix.T)
    return transformed_point_cloud[:, :3]


# In[ ]:


def generate_pointcloud_from_depthmap(u, v, depthmap, intrinsics):
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    C = np.array([cx, cy])
    F = np.array([fx, fy])
    h, w = depthmap.shape
    z = depthmap[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([u,v,z]), np.array([x, y, z])


# In[ ]:


def get_3d_point(argoverse_data, idx, camera, matched_point):
    img = argoverse_data.get_image_sync(idx,camera = camera)
    calib = argoverse_data.get_calibration(camera)
    pc = argoverse_data.get_lidar(idx)
    uv = calib.project_ego_to_image(pc).T
    idx_ = np.where(np.logical_and.reduce((uv[0, :] >= 0.0, uv[0, :] < np.shape(img)[1] - 1.0,
                                                      uv[1, :] >= 0.0, uv[1, :] < np.shape(img)[0] - 1.0,
                                                      uv[2, :] > 0)))
    idx_ = idx_[0]
    uv1 =uv[:, idx_]
    pc1 =pc[idx_, :]
    u = uv1.T
    nearest_index = np.argmin(np.sum(np.abs(u[:,:2]-np.array(matched_point)), axis=1))
    uvd, point_in_3d = u[nearest_index], pc1[nearest_index]
    point_in_3d = argoverse_data.get_pose(idx, log).transform_point_cloud(point_in_3d.reshape(1,-1))
    return uvd, point_in_3d


# In[ ]:


def converter_class(x):
    return x[2:-2]


# In[ ]:


def get_tracks(ob_bbox_path, log, track_file_path, argoverse_data, camera):
    ORIGINAL_HEIGHT = 1024
    ORIGINAL_WIDTH = 1792

    RESIZE_H = 256
    RESIZE_W = 448
    
    object_annotations = pd.read_csv(ob_bbox_path, converters={'class': converter_class})
    
    num_lidar = len(argoverse_data.get_image_list_sync(camera))
    
    trackfile = open(track_file_path, "w")
    header='{},{},{},{},{},{},{},{},{},{},{},{}\n'.format('log', 'frame', 'trackid', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'w_x', 'w_y', 'w_z', 'frame_id')
    trackfile.writelines(header)
    
    global_pose = argoverse_data.get_pose(0, log).transform_matrix
    intrinsics = argoverse_loader.get_calibration(camera, log).camera_config.intrinsic[:3, :3].astype(np.float32)
    
    for frame in range(1, num_lidar):
        
        #Depth
        img_path = argoverse_data.get_image_sync(frame, camera, load=False)
        #img = argoverse_data.get_image_sync(frame, camera, load=True)
        img = imread(img_path).astype(np.float32)[:1024,:1792]
        img = imresize(img, (RESIZE_H, RESIZE_W), anti_aliasing=True).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.45)/0.225).to(device)
        disp = disp_net(tensor_img)
        disp_resized = torch.nn.functional.interpolate(disp,(ORIGINAL_HEIGHT, ORIGINAL_WIDTH), mode="bilinear", align_corners=False)
        depthmap = (1/disp_resized).detach().cpu().squeeze().numpy()
        
        #Pose
        file1 = argoverse_data.get_image_sync(frame, camera, load=False)
        file0 = argoverse_data.get_image_sync(frame-1, camera, load=False)
        img0 = get_resized_img(file0)
        img1 = get_resized_img(file1)
        tensor_img0 = ((img0/255 - 0.45)/0.225).to(device)
        tensor_img1 = ((img1/255 - 0.45)/0.225).to(device)
        pose = pose_net(tensor_img0, tensor_img1)
        #pose_6d.append(pose.squeeze(0).detach().cpu().numpy())
        pose_mat = pose_vec2mat(pose).squeeze(0).detach().cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        global_pose = global_pose @  np.linalg.inv(pose_mat)
        
        #get_3d_point(argoverse_data, frame, camera, matched_point)
        f = os.path.join(object_det_dir, img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1])
        ob_ann = object_annotations[object_annotations.frame == f].values
        for ann in ob_ann:
            #print(ann)
            path_split = ann[1].split('/')
            x_s, y_s, w, h = ann[-4:]
            img_path = os.path.join(root_dir, path_split[-3], path_split[-2], path_split[-1])
            matched_point = x_center, y_center = int(x_s+w/2), int(y_s+h/2)
            if (x_center > ORIGINAL_WIDTH) or (y_center > ORIGINAL_HEIGHT): 
                print(x_center, y_center)
                continue
            #uvd, point_3d = get_3d_point(argoverse_data, frame, camera, matched_point)
            uvd, point_3d = generate_pointcloud_from_depthmap(x_center, y_center, depthmap, intrinsics)
            point = np.ones((4, 1))
            point[0, 0] = point_3d[0]
            point[1, 0] = point_3d[1]
            point[2, 0] = point_3d[2]
            point_3d = point
            point_3d = global_pose@point_3d
            ann = np.append(ann, point_3d.reshape(-1)[:3])
            ann = np.append(ann, np.array([frame]))
            line = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(ann[0],ann[1],ann[2],ann[3],ann[4],ann[5],ann[6],ann[7],ann[8],ann[9],ann[10],ann[11])
            trackfile.writelines(line)
        x, y, z = argoverse_data.get_pose(frame).translation
        line = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(log,ann[1],0,'car',-1,-1,-1,-1,x,y,z,frame)
        trackfile.writelines(line)
    trackfile.close()
    return track_file_path


# In[ ]:


def filter_trajectories(extracted_trajectories, mincount = 10):
    track_ids, counts = np.unique(extracted_trajectories.trackid, return_counts=True)
    filtered_extracted_trajectories = extracted_trajectories[extracted_trajectories.trackid.isin(track_ids[counts > mincount])]
    trajectory_dataframes = []
    for trackid in np.unique(filtered_extracted_trajectories.trackid):
        one_track = filtered_extracted_trajectories[filtered_extracted_trajectories.trackid == trackid]
        classes, counts = np.unique(one_track['class'], return_counts = True)
        max_class_trajectory = one_track[one_track['class']==classes[np.argmax(counts)]]
        trajectory_dataframes.append(max_class_trajectory)
    return pd.concat(trajectory_dataframes)


# In[ ]:


forecasting_file_id = 0
def make_forecasting_traj_files(forecasting_file_id, filtered_extracted_trajectories):
    header='TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME\n'
    for agentid in np.unique(filtered_extracted_trajectories.trackid):
        if agentid == 0:
            continue
        lines = ''
        for trackid in np.unique(filtered_extracted_trajectories.trackid):
            one_track = filtered_extracted_trajectories[filtered_extracted_trajectories.trackid == trackid]
            classes, counts = np.unique(one_track['class'], return_counts = True)
            if trackid == 0:
                obj_type = 'AV'
            elif agentid == trackid:
                obj_type = 'AGENT'
            else:
                obj_type = 'OTHER'
            for record in one_track.values:
                lines+='{},{},{},{},{},{}\n'.format(int(str(timestamps[int(record[-1])])[:10])/10, log+'-'+str(trackid), obj_type, record[-4], record[-3], argoverse_data.city_name)
        forecasting_file_id+=1
        final_trajectory_file = open('{}.csv'.format(str(forecasting_file_id).zfill(4)), "w")
        final_trajectory_file.writelines(header)
        final_trajectory_file.writelines(lines)
        final_trajectory_file.close()
    return forecasting_file_id
    #print(max_class_trajectory.shape)


# In[ ]:


object_det_dir = '/ssd_scratch/cvit/raghava.modhugu/argoverse_tracking/train'
st = time.time()
forecasting_file_id = 0
for log in argoverse_loader.log_list[1:]:
    log = '74750688-7475-7475-7475-474752397312'
    argoverse_data = argoverse_loader.get(log)
    timestamps = list(argoverse_data.timestamp_lidar_dict.keys())
    ob_bbox_path = '{}_detection.csv'.format(log)
    track_file_path = '{}_tracks_bbox.csv'.format(log)
    path = get_tracks(ob_bbox_path, log, track_file_path, argoverse_data, camera)
    extracted_trajectories = pd.read_csv(path)
    filtered_extracted_trajectories = filter_trajectories(extracted_trajectories)
    forecasting_file_id = make_forecasting_traj_files(forecasting_file_id, filtered_extracted_trajectories)
    print(forecasting_file_id)
    break
print(time.time()-st)

