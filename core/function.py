import time
import logging

import numpy as np
import torch
import cv2

from pathlib import Path

from core.inference import get_j2d_from_hms
from core.evaluate import accuracy, root_accuracy
from utils.vis import save_debug_images, save_debug_3d_joints, save_debug_segmenation, save_debug_eros, plot_heatmaps, generate_skeleton_image
from utils.skeleton import Skeleton

logger = logging.getLogger(__name__)


def compute_fn(model, batch, prev_buffer=None, prev_key=None, batch_first=False):
    inps = []

    frame_index = []
    gt_hms = []
    gt_j2d = []
    gt_j3d = []
    gt_seg = []
    vis_j2d = []
    vis_j3d = []
    valid_j3d = []
    valid_seg = []
    for (data, meta) in batch:
        inp = data['x']
        inps.append(inp[None, ...])    

        gt_hms_ = data['hms']
        gt_j3d_ = data['j3d'] 
        gt_seg_ = data['segmentation_mask']

        gt_j2d_ = meta['j2d']

        vis_j2d_ = meta['vis_j2d']
        vis_j3d_ = meta['vis_j3d']
        valid_j3d_ = meta['valid_j3d']
        valid_seg_ = meta['valid_seg']
        frame_index_ = meta['frame_index']
        

        gt_hms.append(gt_hms_)
        gt_j3d.append(gt_j3d_)
        gt_seg.append(gt_seg_)

        gt_j2d.append(gt_j2d_)
        vis_j2d.append(vis_j2d_)
        vis_j3d.append(vis_j3d_)
        valid_j3d.append(valid_j3d_)
        valid_seg.append(valid_seg_)

        frame_index.append(frame_index_)

    del batch

    inps = torch.cat(inps, dim=0).cuda()

    gt_hms = torch.cat(gt_hms, dim=0).cuda()
    gt_j3d = torch.cat(gt_j3d, dim=0).cuda()
    gt_seg = torch.cat(gt_seg, dim=0).cuda()

    gt_j2d = torch.cat(gt_j2d, dim=0).cuda()
    vis_j2d = torch.cat(vis_j2d, dim=0).cuda()
    vis_j3d = torch.cat(vis_j3d, dim=0).cuda()
    valid_j3d = torch.cat(valid_j3d, dim=0).cuda()
    valid_seg = torch.cat(valid_seg, dim=0).cuda()
    frame_index = torch.cat(frame_index, dim=0).cuda()
    
    outputs = model(inps, prev_buffer, prev_key, batch_first)
    
    T, B, C, H, W = inps.shape
    return inps.view(T * B, C, H, W), outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index


def train(config, train_loader, model, criterions, optimizer, epoch, output_dir, tb_log_dir, writer_dict, pretraining=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    hms_losses = AverageMeter()
    j3d_losses = AverageMeter()    
    seg_losses = AverageMeter()

    acc = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        if i > config.TRAIN_ITERATIONS_PER_EPOCH: break

        data_time.update(time.time() - end)
        
        inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index = compute_fn(model, batch)
        meta = {'j3d': gt_j3d, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}   
        
        pred_hms = outputs['hms']
        pred_seg = outputs['seg']
        pred_eros = outputs['eros']
        
        gt_j3d = gt_j3d  * 1000 # scale to mm
        pred_j3d = outputs['j3d'] * 1000 # scale to mm        

        loss_hms = criterions['hms'](pred_hms, gt_hms, vis_j2d * 10)  # scale to 10
        loss_seg = criterions['seg'](pred_seg, gt_seg, valid_seg)
        loss_j3d = criterions['j3d'](pred_j3d, gt_j3d, vis_j3d * 1e-2)  # scale to 1e-2
        
        loss = loss_hms + loss_j3d + loss_seg
  
        pred_j2d = get_j2d_from_hms(config, pred_hms)

        # compute gradient and do update step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        hms_losses.update(loss_hms.item(), inp.size(0))
        seg_losses.update(loss_seg.item(), inp.size(0))
        j3d_losses.update(loss_j3d.item(), inp.size(0))

        losses.update(loss.item(), inp.size(0))
        
        avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'HM_Loss {hms_loss.val:.5f} ({hms_loss.avg:.5f})\t' \
                'J3D_Loss {j3d_loss.val:.5f} ({j3d_loss.avg:.5f})\t' \
                'SEG_Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'MPJPE {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=inp.size(0)/batch_time.val,
                    data_time=data_time, 
                    loss=losses, 
                    j3d_loss=j3d_losses, 
                    hms_loss=hms_losses,
                    seg_loss=seg_losses,
                    acc=acc
                    )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if i % (config.PRINT_FREQ * 4) == 0:
                save_debug_images(config, inp, meta, gt_hms, pred_j2d, pred_hms, 'train', writer, global_steps)
                save_debug_3d_joints(config, inp, meta, gt_j3d, pred_j3d, 'train', writer, global_steps)
                save_debug_segmenation(config, inp, meta, gt_seg, pred_seg, 'train', writer, global_steps)
                save_debug_eros(config, inp, meta, pred_eros, 'train', writer, global_steps)


def validate(config, val_loader, val_dataset, model, criterions, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
        
    acc_j3d = AverageMeter()

    buffer = None
    key = None

    # switch to evaluate mode
    model.eval()

    inp_W, inp_H = config.MODEL.IMAGE_SIZE  
    hm_W, hm_H = config.MODEL.HEATMAP_SIZE

    buffer = torch.zeros(1, 2, inp_H, inp_W).cuda()
    key = torch.ones(1, 1, hm_H, hm_W).cuda()

    all_frame_indices = []
    all_gt_j3ds = []
    all_preds_j3d = []
    all_vis_j3d = []
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):            
            inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index = compute_fn(model, [batch], buffer, key, batch_first=True)                    
            
            meta = {'j3d': gt_j3d, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}   

            pred_hms = outputs['hms']            
            pred_j3d = outputs['j3d'] * 1000 # scale to mm        
            gt_j3d = gt_j3d * 1000 # scale to mm
            
            avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
            acc_j3d.update(avg_acc, cnt)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred_j3d = pred_j3d.detach().cpu().numpy()
            preds_j2d = get_j2d_from_hms(config, pred_hms)
    
            all_preds_j3d.append(pred_j3d)
            all_gt_j3ds.append(gt_j3d.cpu().numpy())
            all_vis_j3d.append(valid_j3d.cpu().numpy())
            all_frame_indices.append(frame_index.cpu().numpy())
            
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        acc=acc_j3d)
                logger.info(msg)

        all_preds_j3d = np.concatenate(all_preds_j3d, axis=0)
        all_gt_j3ds = np.concatenate(all_gt_j3ds, axis=0)
        all_vis_j3d = np.concatenate(all_vis_j3d, axis=0)
        all_frame_indices = np.concatenate(all_frame_indices, axis=0)

        name_values, perf_indicator = val_dataset.evaluate_dataset(config, frame_indices=all_frame_indices, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
                
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
        
        name_values, perf_indicator = val_dataset.evaluate_joints(config, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
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

        save_debug_images(config, inp, meta, gt_hms, preds_j2d, pred_hms, 'test', writer, global_steps)
        save_debug_3d_joints(config, inp, meta, gt_j3d, pred_j3d, 'test', writer, global_steps)


    return perf_indicator

def test(cfg, valid_loader, valid_dataset, model, tb_log_dir, writer_dict, seq_time_in_sec=10):
    fps = 30
    
    seq_len = seq_time_in_sec * fps
    data_len = len(valid_dataset)
    
    global_steps = writer_dict['valid_global_steps']
    np.random.seed(int(global_steps))
    start = np.random.randint(0, data_len - cfg.DATASET.TEMPORAL_STEPS)
    stop = min(start + seq_len, data_len)

    tb_log_dir = Path(tb_log_dir)
    video_path = str(tb_log_dir / f'{global_steps}.mp4')

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (400 * 2, 300 * 3))
    model.eval()
        
    for i in range(start, stop):
        gt_j3d = []
        inps = []
        gt_hms = []
        
        data, meta = valid_dataset[i]

        inp = data['x']
        gt_j3d_ = data['j3d']

        inps.append(inp[None, None, ...])
        gt_j3d.append(gt_j3d_[None, ...])
        gt_hms.append(data['hms'][None, ...])
            
        inps = torch.cat(inps, dim=0).cuda()
        with torch.no_grad():
            outputs = model(inps)

        pred_j3ds = outputs['j3d'].cpu().numpy()
        preds_hms = outputs['hms'].cpu().numpy()
        pred_j2ds = get_j2d_from_hms(cfg, preds_hms)
        
        gt_j3ds = torch.cat(gt_j3d, dim=0).cpu().numpy()
        gt_hms = torch.cat(gt_hms, dim=0).cpu().numpy()
        gt_hm_j2ds = get_j2d_from_hms(cfg, gt_hms)

        T, B, C, H, W = inps.shape
        for i in range(T):
            gt_j3d = gt_j3ds[i]
            gt_hm = gt_hms[i]
            gt_hm_j2d = gt_hm_j2ds[i]
            
            pred_j3d = pred_j3ds[i]
            pred_j2d = pred_j2ds[i]
            pred_hm = preds_hms[i]

            
            inp = valid_dataset.visualize(inps[i, 0])
            pred_hm_image = plot_heatmaps(inp, pred_hm)    
            gt_hm_image = plot_heatmaps(inp, gt_hm)    

            inp_w_gt_hm_j2d = Skeleton.draw_2d_skeleton(inp, gt_hm_j2d, lines=True)
            inp = Skeleton.draw_2d_skeleton(inp, pred_j2d, lines=True)
                    
            color = generate_skeleton_image(gt_j3d, pred_j3d)
            color = color[..., ::-1]
            
            color = cv2.resize(color, (400, 300))
            inp = cv2.resize(inp, (400, 300))
            pred_hm_image = cv2.resize(pred_hm_image, (400, 300))
            gt_hm_image = cv2.resize(gt_hm_image, (400, 300))
            inp_w_gt_hm_j2d = cv2.resize(inp_w_gt_hm_j2d, (400, 300))            
            
            hstack1 = np.concatenate([inp, color], axis=1)
            hstack2 = np.concatenate([gt_hm_image, pred_hm_image], axis=1)
            hstack3 = np.concatenate([inp_w_gt_hm_j2d, np.zeros_like(inp_w_gt_hm_j2d)], axis=1)
            
            vstack = np.concatenate([hstack1, hstack2, hstack3], axis=0)

            video.write(vstack)

    video.release()


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
