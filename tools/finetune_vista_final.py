# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhiqiu Lin
# ------------------------------------------------------------------------------
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
from core.criterion import CrossEntropy, NLLLoss
from core.function import train_all
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, MyModel

SAVE_EPOCHS = [649, 699, 749, 790]

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)  
    # parser.add_argument("--mode", type=str, default='half_1', choices=['half_0', 'half_1', 'upper'])       
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--strategy", type=str, default='none', choices=['none', 'naive', 'filtering', 'conditioning']) 
    parser.add_argument("--samples", type=int, default=2500, choices=[2500, 5000]) 
    parser.add_argument("--loss", type=str, default='joint', choices=['joint', 'lpl']) 
    parser.add_argument("--double_weight", type=bool, default=False) 


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
    
    if args.samples == 2500:
        prev_model_path = os.path.join("output/vista_v1_2/2500_half_0", 'final_state.pth')
        final_output_dir = os.path.join(final_output_dir, f'2500_selftrain_{args.strategy}_loss_{args.loss}_double_{args.double_weight}')
        if args.loss == 'lpl':
            final_output_dir = os.path.join(final_output_dir, f'2500_selftrain_{args.strategy}_loss_{args.loss}_corrected_double_{args.double_weight}')
    elif args.sample == 5000:
        prev_model_path = os.path.join("output/vista_v1_2/half_0", 'final_state.pth')
        final_output_dir = os.path.join(final_output_dir, f'selftrain_{args.strategy}_loss_{args.loss}_double_{args.double_weight}')
    
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
    if local_rank <= 0:
        print(f"Results will be saved at {final_output_dir}")
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)

    # build model
    if args.loss == 'joint':
        model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model_two_head_return_both')(config)
    else:
        model = eval('models.'+config.MODEL.NAME +
                    '.get_seg_model')(config)
    
    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    if args.strategy == 'naive':
        selftrain_list_path = f'samples_{args.samples}_strategy_none.lst'
    else:
        selftrain_list_path = f'samples_{args.samples}_strategy_{args.strategy}.lst'
    t_0_list_path = f"data/list/vista_v1_2_{args.samples*2}/train_half_0_consistent.lst"
    
    t_1_v1_list_path = f'data/list/vista_v1_2_{args.samples*2}/train_half_1.lst'
    t_1_v2_list_path = f'data/list/vista_v2_0_{args.samples*2}/train_half_1.lst'

    
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    t_0_dataset = eval('datasets.'+"vista_both")(
                       root=config.DATASET.ROOT,
                       list_path_0=selftrain_list_path,
                       list_path_1=t_0_list_path,
                       multi_scale=config.TRAIN.MULTI_SCALE,
                       flip=config.TRAIN.FLIP,
                       ignore_label=config.TRAIN.IGNORE_LABEL,
                       base_size=config.TRAIN.BASE_SIZE,
                       crop_size=crop_size,
                       no_label_map=[False, True],
                       downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                       scale_factor=config.TRAIN.SCALE_FACTOR)
    
    t_1_dataset = eval('datasets.'+"vista_both")(
                       root=config.DATASET.ROOT,
                       list_path_0=t_1_v1_list_path,
                       list_path_1=t_1_v2_list_path,
                       multi_scale=config.TRAIN.MULTI_SCALE,
                       flip=config.TRAIN.FLIP,
                       ignore_label=config.TRAIN.IGNORE_LABEL,
                       base_size=config.TRAIN.BASE_SIZE,
                       crop_size=crop_size,
                       no_label_map=[False, False],
                       downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                       scale_factor=config.TRAIN.SCALE_FACTOR)

    
    # print("load train_dataset")
    assert batch_size % 2 == 0
    train_sampler_t_0 = get_sampler(t_0_dataset)
    trainloader_t_0 = torch.utils.data.DataLoader(
        t_0_dataset,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(config.WORKERS/2),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler_t_0)
    
    train_sampler_t_1 = get_sampler(t_1_dataset)
    trainloader_t_1 = torch.utils.data.DataLoader(
        t_1_dataset,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(config.WORKERS/2),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler_t_1)

    # criterion
    if config.LOSS.USE_OHEM:
        import pdb; pdb.set_trace() # NOT IMPLEMENTED
        # criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
        #                                 thres=config.LOSS.OHEMTHRES,
        #                                 min_kept=config.LOSS.OHEMKEEP,
        #                                 weight=train_dataset.class_weights)
    else:
        if args.double_weight:
            weight_vec = torch.Tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0])
            print(f"Wieght vec has shape {weight_vec.shape}")
        else:
            weight_vec = None
        
        criterion_t_0_ce = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                   weight=None)
        criterion_t_1_ce = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                   weight=weight_vec)
        criterion_t_0_nll = NLLLoss(ignore_label=config.TRAIN.IGNORE_LABEL,
                                   weight=None)
    
    # Load last final_state
    pretrained_dict = torch.load(prev_model_path, map_location={'cuda:0': 'cpu'})
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    new_pretrained_dict = {}
    model_dict = model.state_dict()
    if args.loss == 'joint':
        for k, v in pretrained_dict.items():
            if k[6:14] == 'cls_head':
                if local_rank <= 0:
                    logger.info(
                        '=> changing {} from pretrained model'.format(k))
                new_k = k.replace("cls_head", "cls_head_0")
                new_pretrained_dict[new_k[6:]] = v
            elif k[6:14] == 'aux_head':
                new_k = k.replace("aux_head", "aux_head_0")
                if local_rank <= 0:
                    logger.info(
                        '=> changing {} from pretrained model'.format(k))
                new_pretrained_dict[new_k[6:]] = v
            elif k[6:] in model_dict.keys():
                new_pretrained_dict[k[6:]] = v
    else:
        for k, v in pretrained_dict.items():
            if k[6:14] in ['cls_head' ,'aux_head']:
                if local_rank <= 0:
                    logger.info(
                        '=> skipping {} from pretrained model'.format(k))
                continue
            if k[6:] in model_dict.keys():
                new_pretrained_dict[k[6:]] = v
    for k, _ in new_pretrained_dict.items():
        if local_rank <= 0:
            logger.info(
                '=> loading {} from pretrained model'.format(k))
    del pretrained_dict
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    if distributed:
        torch.distributed.barrier()
        
    model = MyModel(model, args.loss, args.strategy,
                    criterion_t_0_nll,
                    criterion_t_0_ce,
                    criterion_t_1_ce)
    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=args.loss == 'joint',
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
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(t_0_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    
    
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
    end_epoch = int(config.TRAIN.END_EPOCH / 2)
    num_iters = end_epoch * epoch_iters
    for epoch in range(last_epoch, end_epoch):
        current_trainloader_t_0 = trainloader_t_0
        current_trainloader_t_1 = trainloader_t_1
        if current_trainloader_t_0.sampler is not None and hasattr(current_trainloader_t_0.sampler, 'set_epoch'):
            current_trainloader_t_0.sampler.set_epoch(epoch)
        if current_trainloader_t_1.sampler is not None and hasattr(current_trainloader_t_1.sampler, 'set_epoch'):
            current_trainloader_t_1.sampler.set_epoch(epoch)

        epoch_time = timeit.default_timer()
        train_all(config, epoch, end_epoch, 
                epoch_iters, config.TRAIN.LR, num_iters,
                trainloader_t_0, trainloader_t_1, optimizer, model, writer_dict)
        if local_rank <= 0:
            print('Minutes for Training: %d' % np.int((-epoch_time+timeit.default_timer())/60))
        
        if local_rank <= 0 and epoch in SAVE_EPOCHS:
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, f'epoch_{epoch}.pth'))
            print("Saved to " + os.path.join(final_output_dir, f'epoch_{epoch}.pth'))
        
        if local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            
            checkpoint_path = os.path.join(final_output_dir,'checkpoint.pth.tar')
            torch.save({
                'epoch': epoch+1,
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
