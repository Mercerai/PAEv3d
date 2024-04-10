import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius): #相机-》世界
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w #
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_data(basedir, half_res=False, testskip=1, splits=None):
    splits = splits
    metas = {}
    with open(os.path.join(basedir, 'transforms_{}.json'.format(splits)), 'r') as fp:
        metas[splits] = json.load(fp)

    all_poses = []
    counts = [0]
    meta = metas[splits]
    poses = []
    if splits=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip

    T = np.array([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    for frame in meta['frames'][::skip]:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        c2w = np.array(frame['transform_matrix'])
        norm = np.sqrt(c2w[0, 3]**2+c2w[1, 3]**2+c2w[2, 3]**2)
        c2w[:3, 3] /= norm
        c2w = c2w @ T
        poses.append(c2w)
    poses = np.array(poses).astype(np.float32)
    all_poses.append(poses)


    poses = np.concatenate(all_poses, 0)

    H, W = [800,800]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # edge_dir = os.path.join(basedir, "edges")
    # edges = []
    # for i in range(200):
    #     fname = os.path.join(edge_dir, "edge_{}.png".format(i))
    #     edge = np.array(imageio.imread(fname), dtype=np.float32)/255.0
    #     # edge = (edge - edge.min())/(edge.max() - edge.min())
    #     edges.append(edge)
    #
    # edges = np.array(edges).astype(np.float32)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    matrix = np.identity(4)
    cx = W/2
    cy = H/2
    matrix[0][0] = focal
    matrix[1][1] = focal
    matrix[0][2] = cx
    matrix[1][2] = cy
    matrix[2][2] = 1
    matrix[3][3] = 1

    return poses, matrix.astype(np.float32)


