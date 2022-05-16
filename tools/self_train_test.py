# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
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

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval_self_train
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--samples", type=int, default=2500, choices=[2500, 5000]) 
    parser.add_argument("--strategy", type=str, default='none', choices=['none', 'filtering', 'conditioning']) 

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    final_output_dir = os.path.join(final_output_dir, f'{args.samples}_selftrain_{args.strategy}')
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED


    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.vista_v2_0')(
                        root=config.DATASET.ROOT,
                        list_path=f"data/list/vista_v2_0_{args.samples*2}/train_half_0.lst",
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    st_dataset = eval('datasets.vista_v2_0')(
                        root=config.DATASET.ROOT,
                        list_path=f'data/list/vista_v2_0_{args.samples*2}/train_half_1_selftrain_{args.strategy}.lst',
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1,
                        no_label_map=True)
    stloader = torch.utils.data.DataLoader(
        st_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    mean_IoU, IoU_array, pixel_acc, mean_acc, mean_pixel_acc = testval_self_train(config, 
                                                                        test_dataset,
                                                                        stloader, 
                                                                        testloader,
                                                                        save_for_self_train=f'samples_{args.samples}_strategy_{args.strategy}'
                                                                        )

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)
    logging.info("Per class pixel acc:")
    logging.info(mean_pixel_acc)
    result = {
        'mean_IoU' : mean_IoU,
        'IoU_array' : IoU_array, 
        'pixel_acc' : pixel_acc, 
        'mean_acc' : mean_acc,
        'mean_pixel_acc' : mean_pixel_acc
    }
    torch.save(result, os.path.join(final_output_dir, "pseudo_label_quality.pt"))
    
    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
