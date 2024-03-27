import utils
import argparse
import os
import pprint
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from rich import print

from settings import config as cfg
from core.loss import HeatMapJointsMSELoss, J3dMSELoss, SegmentationLoss
from core.function import train
from core.function import validate
from core.function import test
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

from dataset import EgoEvent 
from dataset import AugmentedEgoEvent
from dataset import CombinedEgoEvent
from dataset import TemoralWrapper
from model import EgoHPE


def main():
    if cfg.DATASET.TYPE == 'Combined':
        TrainDataset = CombinedEgoEvent
    else:
        TrainDataset = EgoEvent

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, TrainDataset.__name__, 'train')

    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = EgoHPE(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    hms_criterion = HeatMapJointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    j3d_criterion = J3dMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    criterions = {}
    criterions['hms'] = hms_criterion
    criterions['j3d'] = j3d_criterion
    criterions['seg'] = SegmentationLoss().cuda()

    if cfg.DATASET.BG_AUG:
        train_dataset = AugmentedEgoEvent(cfg, TrainDataset(cfg, split='train'))
    else:
        train_dataset = TrainDataset(cfg, split='train')
        
    if cfg.DATASET.BG_AUG:
        finetune_dataset = AugmentedEgoEvent(cfg, EgoEvent(cfg, split='train', finetune=True))
    else:
        finetune_dataset = EgoEvent(cfg, split='train', finetune=True)
    
    cfg.DATASET.TYPE = 'Real'    
    valid_dataset = EgoEvent(cfg, split='test')

    train_dataset = TemoralWrapper(train_dataset, cfg.DATASET.TEMPORAL_STEPS, augment=True)
    finetune_dataset = TemoralWrapper(finetune_dataset, cfg.DATASET.TEMPORAL_STEPS, augment=False)

    batch_size = cfg.BATCH_SIZE * cfg.N_GPUS
    n_workers = 0 if cfg.DEBUG.NO_MP else min(os.cpu_count(), batch_size)

    print(f"BATCH_SIZE: {batch_size}")
    print(f"N_WORKERS: {n_workers}")
    print(f"N_GPUS: {cfg.N_GPUS}")
    print(f'IMAGE_SIZE: {cfg.MODEL.IMAGE_SIZE}')
    print(f'HEATMAP_SIZE: {cfg.MODEL.HEATMAP_SIZE}')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers ,
        pin_memory=True
    )
    
    finetune_loader = torch.utils.data.DataLoader(
        finetune_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers ,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers ,
        pin_memory=True
    )

    best_perf = 1e6
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, cfg.MODEL.CHECKPOINT_PATH
    )
    
    if os.path.isfile(checkpoint_file) and checkpoint_file.endswith('.pth'):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        last_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in trange(begin_epoch, cfg.TRAIN.END_EPOCH, desc='Epoch'):
        try:
            # # train for one epoch
            train(cfg, train_loader, model, criterions, optimizer, epoch, final_output_dir, tb_log_dir, writer_dict, pretraining=True)
            # train(cfg, finetune_loader, model, criterions, optimizer, epoch, final_output_dir, tb_log_dir, writer_dict, pretraining=False)

            # # evaluate on validation set
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterions,
                final_output_dir, tb_log_dir, writer_dict
            )

            test(cfg, valid_loader, valid_dataset, model, tb_log_dir, writer_dict)

            lr_scheduler.step()

        except KeyboardInterrupt as e:
            perf_indicator = 1e6
            
        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint(epoch + 1, {
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, tb_log_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
