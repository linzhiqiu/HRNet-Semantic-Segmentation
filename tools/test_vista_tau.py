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
from core.function import testval, test
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
    parser.add_argument('--tau',
                        help='tau normalization parameter',
                        default=1.0,
                        type=float)
    parser.add_argument("--mode", type=str, default='single_head', choices=['single_head', 'two_head']) 

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    model_state_file = config.TEST.MODEL_FILE
    model_state_folder = model_state_file[:model_state_file.rfind(os.sep)]
    logger.info('=> loading model from {}'.format(model_state_file))
    
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    
    if args.mode == 'single_head':
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            logger.info(
                '=> loading {} from pretrained model'.format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.mode == 'two_head':
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k[6:16] == 'cls_head_1':
                print('=> changing {} from pretrained model'.format(k))
                new_k = k.replace("cls_head_1", "cls_head")
                new_pretrained_dict[new_k[6:]] = v
            elif k[6:16] == 'aux_head_1':
                new_k = k.replace("aux_head_1", "aux_head")
                new_pretrained_dict[new_k[6:]] = v
            elif k[6:] in model_dict.keys():
                new_pretrained_dict[k[6:]] = v
        for k, _ in new_pretrained_dict.items():
            print('=> loading {} from pretrained model'.format(k))
        model_dict.update(new_pretrained_dict)
        del pretrained_dict
        model.load_state_dict(model_dict)

    norms = torch.linalg.norm(model.cls_head.weight.data, ord=2, dim=1).squeeze()
    sorted_indices = [58,20,61,26,23,102,113,60,4,81,11,3,42,25,18,114,115,34,83,108,59,12,75,2,29,24,33,94,13,22,101,5,93,56,16,79,112,47,15,46,87,21,82,19,62,14,48,111,51,63,28,104,69,71,99,80,91,98,106,30,45,8,66,86,105,31,50,17,9,77,10,92,49,95,64,41,53,76,7,74,57,78,84,107,73,36,88,97,100,43,103,39,109,40,37,38,72,90,68,65,70,1,67,35,27,89,85,110,96,44,55,0,54,52,32,6]
    import pdb; pdb.set_trace()
    model.cls_head.weight.data = model.cls_head.weight.data / (torch.linalg.norm(model.cls_head.weight.data, ord=2, dim=1).unsqueeze(1).detach() ** args.tau)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
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
    
    start = timeit.default_timer()
    print(f"Saving result to {model_state_folder}")
    import pdb; pdb.set_trace()
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
                                                        test_dataset, 
                                                        testloader, 
                                                        model)

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)
    
    result = {
        'mean_IoU' : mean_IoU,
        'IoU_array' : IoU_array, 
        'pixel_acc' : pixel_acc, 
        'mean_acc' : mean_acc,
    }
    torch.save(result, os.path.join(model_state_folder, f"test_result_tau_{args.tau}.pt"))
    # elif 'test' in config.DATASET.TEST_SET:
    #     test(config, 
    #          test_dataset, 
    #          testloader, 
    #          model,
    #          sv_dir=final_output_dir)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
