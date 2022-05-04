# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhiqiu Lin
# ------------------------------------------------------------------------------
# torchrun --standalone --nnodes=1 --nproc_per_node=4 tools/train_vista_two_head.py --cfg experiments/vista_v1_2/v1_2_2090.yaml
import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train_two_head
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModelTwoHead

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)       
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None

def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    
    prev_model_path = os.path.join("output/vista_v1_2/v1_2_3090_half_0_200_epochs", 'final_state.pth')
    # prev_model_path = os.path.join(final_output_dir, 'final_state.pth') #TODO:
    final_output_dir = os.path.join(final_output_dir, 'two_head')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    local_rank = int(os.environ["LOCAL_RANK"])
    distributed = local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model_two_head')(config)
    
    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    
    if distributed:
        test_batch_size = config.TEST.BATCH_SIZE_PER_GPU
    else:
        test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(gpus)

    list_path_t_0 = 'data/list/vista_v1_2/train_half_0.lst'
    list_path_t_1 = 'data/list/vista_v2_0/train_half_1.lst'
    
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset_t_0 = eval('datasets.'+"vista_v1_2")(
                            root=config.DATASET.ROOT,
                            list_path=list_path_t_0,
                            multi_scale=config.TRAIN.MULTI_SCALE,
                            flip=config.TRAIN.FLIP,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            base_size=config.TRAIN.BASE_SIZE,
                            crop_size=crop_size,
                            class_balancing=None,
                            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                            scale_factor=config.TRAIN.SCALE_FACTOR)
    train_dataset_t_1 = eval('datasets.'+"vista_v2_0")(
                            root=config.DATASET.ROOT,
                            list_path=list_path_t_1,
                            multi_scale=config.TRAIN.MULTI_SCALE,
                            flip=config.TRAIN.FLIP,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            base_size=config.TRAIN.BASE_SIZE,
                            crop_size=crop_size,
                            class_balancing=None,
                            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                            scale_factor=config.TRAIN.SCALE_FACTOR)
    # print("load train_dataset")
    assert batch_size % 2 == 0
    train_sampler_t_0 = get_sampler(train_dataset_t_0)
    trainloader_t_0 = torch.utils.data.DataLoader(
        train_dataset_t_0,
        batch_size=int(batch_size / 2),
        shuffle=False,
        # num_workers=config.WORKERS,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler_t_0)

    train_sampler_t_1 = get_sampler(train_dataset_t_1)
    trainloader_t_1 = torch.utils.data.DataLoader(
        train_dataset_t_1,
        batch_size=int(batch_size / 2),
        shuffle=False,
        # num_workers=config.WORKERS,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler_t_1)

    # criterion
    if config.LOSS.USE_OHEM:
        import pdb; pdb.set_trace() # NOT TESTED
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        criterion_t_0 = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset_t_0.class_weights)
        criterion_t_1 = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset_t_1.class_weights)

    model = FullModelTwoHead(model, criterion_t_0, criterion_t_1)
    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            # find_unused_parameters=True,
            device_ids=[local_rank],
            output_device=local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR}, {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR, #TODO: Tune it
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset_t_1.__len__() * 2 / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    
    
    # Load last final_state
    pretrained_dict = torch.load(prev_model_path)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if distributed:
        torch.distributed.barrier()

    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = int(config.TRAIN.END_EPOCH / 2) # TODO: Make sure this is correct
    num_iters = end_epoch * epoch_iters
    for epoch in range(last_epoch, end_epoch):
        current_trainloader_t_0 = trainloader_t_0
        current_trainloader_t_1 = trainloader_t_1
        if current_trainloader_t_0.sampler is not None and hasattr(current_trainloader_t_0.sampler, 'set_epoch'):
            current_trainloader_t_0.sampler.set_epoch(epoch)
        if current_trainloader_t_1.sampler is not None and hasattr(current_trainloader_t_1.sampler, 'set_epoch'):
            current_trainloader_t_1.sampler.set_epoch(epoch)

        epoch_time = timeit.default_timer()
        train_two_head(config, epoch, config.TRAIN.END_EPOCH, 
                epoch_iters, config.TRAIN.LR, num_iters,
                trainloader_t_0, trainloader_t_1, optimizer, model, writer_dict)
        if local_rank <= 0:
            print('Minutes for Training: %d' % np.int((-epoch_time+timeit.default_timer())/60))
        
        if local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            
            checkpoint_path = os.path.join(final_output_dir,'checkpoint.pth.tar')
            torch.save({
                'epoch': epoch+1,
                # 'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, checkpoint_path)
            print(f"Saving to {checkpoint_path}")

        if distributed:
            torch.distributed.barrier()
    
    if local_rank <= 0:

        torch.save(model.module.state_dict(),
                os.path.join(final_output_dir, 'final_state.pth'))
        print("Saved to " + os.path.join(final_output_dir, 'final_state.pth'))
        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end-start)/3600))
        logger.info('Done')
    

if __name__ == '__main__':
    main()
