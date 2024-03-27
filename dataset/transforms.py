# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import ocam


def flip_lr(img):
    return np.fliplr(img).copy()

def flip_lr_joints(inp, j2d):
    assert j2d.ndim == 2, 'j2d should be [num_joints, 2]'
    h, w = inp.shape[:2]
    j2d = j2d.copy()
    
    j2d[:, 0] = w - j2d[:, 0] - 1
    
    return j2d
    ...

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform_pts(pts, t):
    if len(pts.shape) == 1:
        pts = pts[None, ...]

    pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    trans_pts = np.matmul(t, pts.T).T
    
    return trans_pts    

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, A, output_size):
    dst_img = cv2.warpAffine(img,
                             A,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_AREA)

    return dst_img


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def flip_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    
    return x.copy()

def random_dropout(x, p):
    mask = np.random.rand(*x.shape) > p
    return x * mask


def rotate_points(points, axis, angle_degrees):
    """
    Rotate a batch of points by a specified angle (in degrees) around a given axis.

    Args:
        points (numpy.ndarray): The batch of points to rotate, where each row represents a point.
        axis (str): The axis of rotation ('x', 'y', or 'z').
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: The rotated batch of points.
    """
    angle_rad = np.radians(angle_degrees)  # Convert angle to radians

    # Create the appropriate rotation matrix based on the axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    # Rotate the batch of points
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points

def camera_to_j2d_batch(gt_j3d, config):
    ocam_model = config.ocam_model
    
    h, w = ocam_model['height'], ocam_model['width']
    gt_j3d = gt_j3d.clone()
    gt_j3d[:, :, 2] *= -1

    point_2Ds = ocam.world2cam_torch_batch(gt_j3d[:, :, None, :], ocam_model)[:, :, 0, :]
    point_2Ds[:, :, 1] = h - point_2Ds[:, :, 1]
    
    width, height = config.MODEL.IMAGE_SIZE   
    sx = width / w
    sy = height / h

    point_2Ds[:, :, 0] *= sx
    point_2Ds[:, :, 1] *= sy
    
    return point_2Ds