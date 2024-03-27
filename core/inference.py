# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from dataset.transforms import transform_preds
import torch

def get_max_preds_torch(batch_heatmaps):
    '''
    get predictions from score maps
    batch_heatmaps: torch.Tensor([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, torch.Tensor), \
        'batch_heatmaps should be torch.Tensor'
    assert batch_heatmaps.dim() == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, height, width = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.view(batch_size, num_joints, -1)
    idx = torch.argmax(heatmaps_reshaped, dim=2)
    maxvals = torch.amax(heatmaps_reshaped, dim=2)

    maxvals = maxvals.view(batch_size, num_joints, 1)
    idx = idx.view(batch_size, num_joints, 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = torch.floor(preds[:, :, 1] / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2).float()

    preds *= pred_mask
    return preds, maxvals


def get_max_preds_np(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_preds(batch_heatmaps):
    if isinstance(batch_heatmaps, torch.Tensor):
        return get_max_preds_torch(batch_heatmaps)
    elif isinstance(batch_heatmaps, np.ndarray):
        return get_max_preds_np(batch_heatmaps)
    else:
        raise TypeError('batch_heatmaps should be torch.Tensor or numpy.ndarray')


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def get_j2d_from_hms(cfg, batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3] 

    width, height = cfg.MODEL.IMAGE_SIZE

    coords[:, :, 0] = coords[:, :, 0] * width / heatmap_width
    coords[:, :, 1] = coords[:, :, 1] * height / heatmap_height

    return coords

def get_j3d_from_lms(cfg, j2d, batch_lms):
    B, J, C, H, W = batch_lms.shape

    width, height = cfg.MODEL.IMAGE_SIZE

    heatpoints = torch.zeros(B, J, 2).to(batch_lms.device)
    
    heatpoints[:, :, 0] = j2d[:, :, 0] * W / width
    heatpoints[:, :, 1] = j2d[:, :, 1] * H / height

    j3d = torch.zeros(B, J, 3).to(batch_lms.device)
    for bdx in range(B):
        for jdx in range(J):
            hx = heatpoints[bdx, jdx, 0].long()
            hy = heatpoints[bdx, jdx, 1].long()

            j3d[bdx, jdx] = batch_lms[bdx, jdx, :, hy, hx]
        
    return j3d