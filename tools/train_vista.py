# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
# torchrun --standalone --nnodes=1 --nproc_per_node=4 tools/train_vista.py --cfg experiments/vista_v1_2/v1_2_2090.yaml
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
from core.function import train, validate_vista
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

SAVE_EPOCHS = [399, 499, 599, 699, 799, 899, 999, 1099, 1199, 1299]

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
                 '.get_seg_model')(config)
    
    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # copy model file
    if distributed and local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        # if os.path.exists(models_dst_dir):
        #     shutil.rmtree(models_dst_dir)
        # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)
        # print(f"copy to {this_dir} with models_dst_dir being {models_dst_dir}")
    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    
    if distributed:
        test_batch_size = config.TEST.BATCH_SIZE_PER_GPU
    else:
        test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        class_balancing=config.TRAIN.CLASS_INVERSE_WEIGHTING,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)
    # print("load train_dataset")
    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    # test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # print(test_size)
    # val_dataset = eval('datasets.'+config.DATASET.DATASET)(
    #                     root=config.DATASET.ROOT,
    #                     list_path=config.DATASET.VAL_SET,
    #                     multi_scale=False,
    #                     flip=False,
    #                     ignore_label=config.TRAIN.IGNORE_LABEL,
    #                     base_size=config.TEST.BASE_SIZE,
    #                     crop_size=test_size,
    #                     downsample_rate=1)
    # assert len(val_dataset) & len(gpus) == 0
    # val_sampler = get_sampler(val_dataset)
    # valloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=test_batch_size,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True,
    #     sampler=val_sampler)
    # test_dataset = eval('datasets.'+config.DATASET.DATASET)(
    #                     root=config.DATASET.ROOT,
    #                     list_path=config.DATASET.TEST_SET,
    #                     multi_scale=False,
    #                     flip=False,
    #                     ignore_label=config.TRAIN.IGNORE_LABEL,
    #                     base_size=config.TEST.BASE_SIZE,
    #                     crop_size=test_size,
    #                     downsample_rate=1)
    
    # test_sampler = get_sampler(test_dataset)
    # assert len(test_dataset) & len(gpus) == 0
    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=test_batch_size,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True,
    #     sampler=test_sampler)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    model = FullModel(model, criterion)
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
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    
    # best_mIoU = 0
    # val_best_mIoU = 0
    # best_epoch = 0
    # val_mIoUs = []
    # test_mIoUs = []
    # val_IoU_arrays = []
    # test_IoU_arrays = []
    # val_pixel_accs = []
    # test_pixel_accs = []
    # val_mean_accs = []
    # test_mean_accs = []
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            # best_mIoU = checkpoint['best_mIoU']
            # val_best_mIoU = checkpoint['val_best_mIoU']
            # best_epoch = checkpoint['best_epoch']
            # val_mIoUs = checkpoint['val_mIoUs']
            # test_mIoUs = checkpoint['test_mIoUs']
            # val_IoU_arrays = checkpoint['val_IoU_arrays']
            # test_IoU_arrays = checkpoint['test_IoU_arrays']
            # val_pixel_accs = checkpoint['val_pixel_accs']
            # test_pixel_accs = checkpoint['test_IoU_arrays']
            # val_mean_accs = checkpoint['val_mean_accs']
            # test_mean_accs = checkpoint['test_mean_accs']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    for epoch in range(last_epoch, end_epoch):
        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        epoch_time = timeit.default_timer()
        train(config, epoch, config.TRAIN.END_EPOCH, 
                epoch_iters, config.TRAIN.LR, num_iters,
                trainloader, optimizer, model, writer_dict)
        if local_rank <= 0:
            print('Minutes for Training: %d' % np.int((-epoch_time+timeit.default_timer())/60))
        
        # epoch_time = timeit.default_timer()
        # valid_loss, val_mean_IoU, val_IoU_array, val_pixel_acc, val_mean_acc = validate_vista(config, 
        #             valloader, model, writer_dict, phase='valid')
        
        # if local_rank <= 0:
        #     print('Minutes for Validation: %d' % np.int((-epoch_time+timeit.default_timer())/60))
        
        # test_loss, test_mean_IoU, test_IoU_array, test_pixel_acc, test_mean_acc = validate_vista(config, 
        #             testloader, model, writer_dict, phase='test')

        if local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            # if val_mean_IoU > val_best_mIoU:
            #     val_best_mIoU = val_mean_IoU
            #     best_epoch = epoch
            #     torch.save(model.module.state_dict(),
            #             os.path.join(final_output_dir, 'best_val.pth'))
            #     print("Saved to " + os.path.join(final_output_dir, 'best_val.pth'))

            # val_mIoUs.append(val_mean_IoU)
            # test_mIoUs.append(test_mean_IoU)
            # val_IoU_arrays.append(val_IoU_array)
            # test_IoU_arrays.append(test_IoU_array)
            # val_pixel_accs.append(val_pixel_acc)
            # test_pixel_accs.append(test_pixel_acc)
            # val_mean_accs.append(val_mean_acc)
            # test_mean_accs.append(test_mean_acc)
            
            checkpoint_path = os.path.join(final_output_dir,'checkpoint.pth.tar')
            torch.save({
                'epoch': epoch+1,
                # 'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'val_best_mIoU': val_best_mIoU,
                # 'best_epoch': best_epoch,
                # 'val_mIoUs': val_mIoUs,
                # 'test_mIoUs': test_mIoUs,
                # 'val_IoU_arrays': val_IoU_arrays,
                # 'test_IoU_arrays': test_IoU_arrays,
                # 'val_pixel_accs': val_pixel_accs,
                # 'test_pixel_accs': test_pixel_accs,
                # 'val_mean_accs': val_mean_accs,
                # 'test_mean_accs': test_mean_accs,
                }, checkpoint_path)
            print(f"Saving to {checkpoint_path}")
            # msg = 'Best Val mIoU: {: 4.4f}, Best Test mIoU: {: 4.4f}. Best Test mIoU (best val epoch {}): {: 4.4f}'.format(
            #             val_best_IoU, max(test_mIoUs), best_epoch, test_mIoUs[best_epoch])
            # logging.info(msg)
            # logging.info(IoU_array)

        if local_rank <= 0 and epoch in SAVE_EPOCHS:
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, f'epoch_{epoch}.pth'))
            print("Saved to " + os.path.join(final_output_dir, f'epoch_{epoch}.pth'))
            
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
    
        # model_state_file = os.path.join(final_output_dir, 'best_val.pth')
        # # model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        # assert os.path.isfile(model_state_file)
        # checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
        # print("Loaded!")
        # model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model.')})
        # mean_IoU, IoU_array, pixel_acc, mean_acc = validate_vista(config, val_dataset, valloader, model, writer_dict=None, phase='valid')
        # print(f"This best val model achieves {mean_IoU} val IoU (Best val IoU is {best_mIoU}) and pixel_acc {pixel_acc} and mean_acc {mean_acc}")
        # test_mean_IoU, test_IoU_array, test_pixel_acc, test_mean_acc = validate_vista(config, test_dataset, testloader, model, writer_dict=None, phase='test')
        # print(f"This best val model achieves {test_mean_IoU} test IoU (Best test IoU is ?) and test_pixel_acc {test_pixel_acc} and test_mean_acc {test_mean_acc}")

    


if __name__ == '__main__':
    main()
