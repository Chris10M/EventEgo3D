import utils
import pprint
import logging
import os
import sys
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import time
import numpy as np
from collections import defaultdict
from tqdm import trange
from rich import print
from torch.utils.tensorboard import SummaryWriter

from model import EgoHPE
from settings import config as cfg
from core.loss import HeatMapJointsMSELoss, J3dMSELoss, SegmentationLoss
from core.function import test, accuracy, root_accuracy, AverageMeter, _print_name_value
from dataset import EgoEvent, TemoralWrapper
from utils.utils import get_logger


def load_model_info(cfg):    
    cfg.DATASET.TYPE = 'Real'
    cfg.DATASET.BG_AUG = False

    model_path = 'saved_models/best_model_state_dict.pth'

    return {
        'model': EgoHPE,
        'path': model_path,
        'name': 'EE3D'
    }
    
logger = logging.getLogger(__name__)

def validate(config, val_loader, val_dataset, model, criterions, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
        
    acc_j3d = AverageMeter()
    len_dataset = len(val_dataset)

    print(f'Length of dataset: {len_dataset}')
    # switch to evaluate mode
    model.eval()

    all_gt_j3ds = []
    all_preds_j3d = []
    all_vis_j3d = []
    all_frame_indices = []
    
    inps = []
    gt_j3d = []
    valid_j3d = []
    valid_seg = []
    gt_hms = []
    frame_indices = []

    end = time.time()
    for i, (data, meta) in enumerate(val_dataset, 1):                
        inp = data['x']

        inps.append(inp[None, None, ...])
        gt_j3d.append(data['j3d'][None, ...])
        gt_hms.append(data['hms'][None, ...])

        frame_indices.append(meta['frame_index'])
        
        valid_j3d.append(meta['valid_j3d'][None, ...])
        valid_seg.append(torch.tensor(meta['valid_seg'])[None, ...])

        if i % cfg.DATASET.TEMPORAL_STEPS == 0:
            inps = torch.cat(inps, dim=0).cuda()
            with torch.no_grad():
                outputs = model(inps)

            gt_j3d = torch.cat(gt_j3d, dim=0)
            gt_hms = torch.cat(gt_hms, dim=0)
            valid_j3d = torch.cat(valid_j3d, dim=0)
            valid_seg = torch.cat(valid_seg, dim=0)
                        
            pred_j3d = outputs['j3d'] * 1000 # scale to mm        
            gt_j3d = gt_j3d * 1000 # scale to mm
            
            avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
            acc_j3d.update(avg_acc, cnt)
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            all_preds_j3d.append(pred_j3d.cpu().numpy())
            all_gt_j3ds.append(gt_j3d.cpu().numpy())
            all_vis_j3d.append(valid_j3d.cpu().numpy())
            all_frame_indices.extend(frame_indices)
                        
            inps = []
            gt_j3d = []
            valid_j3d = []
            valid_seg = []
            gt_hms = []
            frame_indices = []
            outputs = defaultdict(list)
                    
        if i % config.PRINT_FREQ == 0:
            msg = 'Test: [{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                    i, len_dataset, batch_time=batch_time,
                    acc=acc_j3d,
                    )
            logger.info(msg)

    all_preds_j3d = np.concatenate(all_preds_j3d, axis=0)
    all_gt_j3ds = np.concatenate(all_gt_j3ds, axis=0)
    all_vis_j3d = np.concatenate(all_vis_j3d, axis=0)
    all_frame_indices = np.array(all_frame_indices)

    name_values, perf_indicator = val_dataset.evaluate_dataset(config, frame_indices=all_frame_indices, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
            
    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar(
        'valid_acc_j3d',
        acc_j3d.avg,
        global_steps
    )

    for key, value in name_values.items():
        writer.add_scalar(
            f'valid_{key}',
            value,
            global_steps
        )
    writer_dict['valid_global_steps'] = global_steps + 1

    writer = writer_dict['writer']

    return perf_indicator


def main():
    model_info = load_model_info(cfg)

    model_path = model_info['path']
    model = model_info['model'](cfg)
    exp_name = model_info['name']

    logger, final_output_dir, tb_log_dir = get_logger(cfg, exp_name, 'train')
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    logger.info(cfg)
    pprint.pprint(cfg)
    
    if not os.path.exists(model_path):
        raise Exception(f"Model path {model_path} does not exist")

    print(f'Loading model from [red]{model_path}')
    checkpoint = torch.load(model_path)
    if 'best_state_dict' in checkpoint:
        checkpoint = checkpoint['best_state_dict']

    model.load_state_dict(checkpoint, strict=False)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # define loss function (criterion) and optimizer
    hms_criterion = HeatMapJointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    j3d_criterion = J3dMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    criterions = {}
    criterions['hms'] = hms_criterion
    criterions['j3d'] = j3d_criterion
    
    test_dataset_e = EgoEvent(cfg, split='test')
    test_dataset = TemoralWrapper(test_dataset_e, cfg.DATASET.TEMPORAL_STEPS, augment=False)

    batch_size = cfg.BATCH_SIZE * cfg.N_GPUS
    n_workers = 0 if cfg.DEBUG.NO_MP else min(16, batch_size)

    print(f"BATCH_SIZE: {batch_size}")
    print(f"N_WORKERS: {n_workers}")
    print(f"N_GPUS: {cfg.N_GPUS}")
    print(f'IMAGE_SIZE: {cfg.MODEL.IMAGE_SIZE}')
    print(f'HEATMAP_SIZE: {cfg.MODEL.HEATMAP_SIZE}')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers ,
        pin_memory=True
    )

    perf_indicator = validate(
        cfg, test_loader, test_dataset_e, model, criterions,
        None, tb_log_dir, writer_dict
    )

    for _ in range(10):
        test(cfg, test_loader, test_dataset_e, model, tb_log_dir, writer_dict, seq_time_in_sec=20)
        writer_dict['valid_global_steps'] += 1

if __name__ == '__main__':
    main()
