# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from PIL import Image
import utils.distributed as dist


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

def train_multi_head(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
                   trainloader_t_0, trainloader_t_1, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, (batch_t_0, batch_t_1) in enumerate(zip(trainloader_t_0, trainloader_t_1)):
        images_t_0, labels_t_0, _, _ = batch_t_0
        images_t_1, labels_t_1, _, _ = batch_t_1
        
        # images = torch.cat((images_t_0, images_t_1), 0)
        # images = images.cuda()
        images_t_0 = images_t_0.cuda()
        images_t_1 = images_t_1.cuda()
        labels_t_0 = labels_t_0.long().cuda()
        labels_t_1 = labels_t_1.long().cuda()
        
        # counter = 0
        # time_indices_0 = []
        # time_indices_1 = []
        # for i in range(images_t_0.shape[0]):
        #     time_indices_0.append(counter)
        #     counter += 1
        # for i in range(images_t_1.shape[0]):
        #     time_indices_1.append(counter)
        #     counter += 1
        loss, _, _ = model(images_t_0, images_t_1, labels_t_0, labels_t_1)
        loss = loss.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            # if idx % 10 == 0:
            #     print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

def validate_vista(config, loader, model, writer_dict=None, phase='valid'):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
                
            x = pred[config.TEST.OUTPUT_INDEX]
            x = F.interpolate(
                input=x, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            confusion_matrix += get_confusion_matrix(
                label,
                x,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL
            )

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    if dist.get_rank() <= 0:
        logging.info('IoU_array={} mean_IoU={} pixel_acc={} mean_acc={}'.format(IoU_array, mean_IoU, pixel_acc, mean_acc))
    
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar(f'{phase}_loss', ave_loss.average(), global_steps)
    writer.add_scalar(f'{phase}_mIoU', mean_IoU, global_steps)
    writer_dict[f'{phase}_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, pixel_acc, mean_acc

# def validate_vista(config, test_dataset, testloader, model, writer_dict=None, phase='valid'):
#     model.eval()
#     confusion_matrix = np.zeros(
#         (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
#     with torch.no_grad():
#         for idx, batch in enumerate(tqdm(testloader)):
#             image, label, _, _ = batch
#             size = label.size()
#             image = image.cuda()
#             label = label.long().cuda()
#             pred = test_dataset.multi_scale_inference(
#                 config,
#                 model,
#                 image,
#                 scales=config.TEST.SCALE_LIST,
#                 flip=config.TEST.FLIP_TEST
#             )

#             if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
#                 pred = F.interpolate(
#                     pred, size[-2:],
#                     mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
#                 )

#             confusion_matrix += get_confusion_matrix(
#                 label,
#                 pred,
#                 size,
#                 config.DATASET.NUM_CLASSES,
#                 config.TRAIN.IGNORE_LABEL)

#     # if dist.is_distributed():
#     #     confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
#     #     reduced_confusion_matrix = reduce_tensor(confusion_matrix)
#     #     confusion_matrix = reduced_confusion_matrix.cpu().numpy()

#     pos = confusion_matrix.sum(1)
#     res = confusion_matrix.sum(0)
#     tp = np.diag(confusion_matrix)
#     pixel_acc = tp.sum()/pos.sum()
#     mean_acc = (tp/np.maximum(1.0, pos)).mean()
#     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#     mean_IoU = IoU_array.mean()

#     if writer_dict:
#         writer = writer_dict['writer']
#         global_steps = writer_dict['valid_global_steps']
#         writer.add_scalar(f'{phase}_mIoU', mean_IoU, global_steps)
#         if phase == 'valid':
#             writer_dict['valid_global_steps'] = global_steps + 1
#     return mean_IoU, IoU_array, pixel_acc, mean_acc
def write_lst(file_name, lst):
    with open(file_name, 'w+') as file:
        for line in lst:
            file.write(line + "\n")

def save_self_train_pred(pred, label, pred_path, edge_matrix, strategy):
    with torch.no_grad():
        pred = pred.to(edge_matrix.device)
        label = label.to(edge_matrix.device)
        filter_mask = label == 255
        if strategy == 'none':
            pred_max = pred.argmax(dim=1)
        elif strategy == 'filtering':
            pred_max = pred.argmax(dim=1)
            pred_one_hot = torch.nn.functional.one_hot(pred_max, num_classes=edge_matrix.shape[0])
            pred_one_hot = pred_one_hot.float().matmul(edge_matrix.float())
            pred_v1_max = pred_one_hot.argmax(3)
            filter_mask = torch.logical_or(filter_mask, pred_v1_max != label)
        elif strategy == 'conditioning':
            pred = pred - pred.min()
            label_mask = edge_matrix.T[label.long()].permute(0, 3, 1, 2)
            pred = pred * label_mask
            pred_max = pred.argmax(dim=1)
        else:
            raise NotImplementedError() 
        pred_max[filter_mask] = 255
        pred_numpy = pred_max.squeeze().cpu().numpy()
        pred = pred.cpu()
        label = label.cpu()
        pred_max = pred_max.cpu()
    im = Image.fromarray(pred_numpy.astype(np.uint8))
    im.save(pred_path)

def self_train_hard(config, test_dataset, testloader, model,
                    save_for_self_train="", train_dir_path='training/images/', label_dir_path='training/v1.2/',
                    root='/project_data/ramanan/zhiqiu/mapillary_vista/',
                    edge_matrix=None, strategy='none'):
    if len(save_for_self_train) > 0:
        lst_save_path = f"{save_for_self_train}.lst"
        if strategy == 'none':
            pass
        elif strategy == 'conditioning':
            assert type(edge_matrix) != type(None)
        elif strategy == 'filtering':
            assert type(edge_matrix) != type(None)
        else:
            raise NotImplementedError()
        print(f"Saving images at {root}/{label_dir_path}/{save_for_self_train}")
        print(f"Lst saved at {lst_save_path}")
        if not os.path.exists(os.path.join(root, label_dir_path, save_for_self_train)):
            os.makedirs(os.path.join(root, label_dir_path, save_for_self_train))
        lst = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)
            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            if len(save_for_self_train) > 0:
                train_path = os.path.join(train_dir_path, name[0]+".jpg")
                pred_path = os.path.join(label_dir_path, save_for_self_train, name[0]+".png")
                lst.append(f"{train_path} {pred_path}")
                save_self_train_pred(
                    pred,
                    label,
                    os.path.join(root, pred_path),
                    edge_matrix,
                    strategy
                )

    if len(save_for_self_train) > 0:
        write_lst(lst_save_path, lst)
        print(f"Lst saved at {lst_save_path}")

def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False, ):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)
            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
    
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum() 
    mean_pixel_acc = tp/np.maximum(1.0, pos)
    mean_acc = mean_pixel_acc.mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc, pixel_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
                
def testval_self_train(config, testdataset, selftrainloader, testloader,
                       save_for_self_train="", label_dir_path='training/v1.2/',
                    root='/project_data/ramanan/zhiqiu/mapillary_vista/'):
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, (batch_st, batch_t) in enumerate(tqdm(zip(selftrainloader, testloader))):
            _, label_t, _, name, *border_padding = batch_t
            _, label_st, _, _, *border_padding = batch_st
            size = label_t.size()
            pred_path = os.path.join(root, label_dir_path, save_for_self_train, name[0]+".png")
            label_st_converted = testdataset.convert_label(label_st.clone().numpy(), inverse=True)
            im = Image.fromarray(label_st_converted.squeeze().astype(np.uint8))
            im.save(pred_path)
            label_st_ignore = label_st == config.TRAIN.IGNORE_LABEL
            label_st[label_st_ignore] = 0
            label_st = torch.nn.functional.one_hot(label_st.long(), num_classes=116).permute(0, 3, 1, 2)
            confusion_matrix += get_confusion_matrix(
                label_t,
                label_st,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                pixel_acc = tp.sum()/pos.sum() 
                mean_pixel_acc = tp/np.maximum(1.0, pos)
                mean_acc = mean_pixel_acc.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
                logging.info('mean_acc: %.4f' % (mean_acc))
                logging.info('pixel_acc: %.4f' % (pixel_acc))
    
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum() 
    mean_pixel_acc = tp/np.maximum(1.0, pos)
    mean_acc = mean_pixel_acc.mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc, pixel_acc

