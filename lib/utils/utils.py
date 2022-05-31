# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs

class FullModelTwoHead(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss_0, loss_1):
    super(FullModelTwoHead, self).__init__()
    self.model = model
    self.loss_0 = loss_0
    self.loss_1 = loss_1

  def forward(self, inputs_t_0, inputs_t_1, labels_t_0, labels_t_1):
    outputs_0 = self.model(inputs_t_0, time=0)
    outputs_1 = self.model(inputs_t_1, time=1)
    # outputs_0 = [outputs_0[i][time_indices_0] for i in range(len(outputs_0))]
    # for i in range(len(outputs_0)):
    #     print(f"outputs_0 has shape at idx {i} being {outputs_0[i].shape}")
    # outputs_1 = [outputs_1[i][time_indices_1] for i in range(len(outputs_1))]
    # for i in range(len(outputs_1)):
    #     print(f"outputs_1 has shape at idx {i} being {outputs_1[i].shape}")
    loss_0 = self.loss_0(outputs_0, labels_t_0)
    loss_1 = self.loss_1(outputs_1, labels_t_1)
    loss = loss_0 + loss_1
    return torch.unsqueeze(loss,0), outputs_0, outputs_1

class FullModelSingleHead(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss_0, loss_1, edge_matrix_path="./edge_matrix_116.pt"):
    super(FullModelSingleHead, self).__init__()
    self.model = model
    self.loss_0 = loss_0 # this is nll loss instead of cross entropy loss
    self.loss_1 = loss_1 # this is nll loss instead of cross entropy loss
    self.edge_matrix = torch.load(edge_matrix_path).float().cuda()
    self.log_softmax = torch.nn.LogSoftmax(dim=1)
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, inputs_t_0, inputs_t_1, labels_t_0, labels_t_1):
    outputs_0 = self.model(inputs_t_0)
    outputs_1 = self.model(inputs_t_1)
    
    for i in range(len(outputs_0)):
      outputs_0[i] = outputs_0[i] - outputs_0[i].max(1)[0].unsqueeze(1)
      outputs_0[i] = self.softmax(outputs_0[i])
      outputs_0[i] = torch.log(torch.matmul(outputs_0[i].transpose(1,3), self.edge_matrix).transpose(1,3)  + 1e-20)
    outputs_1 = [self.log_softmax(outputs_1[i]) for i in range(len(outputs_1))]
    loss_0 = self.loss_0(outputs_0, labels_t_0)
    # loss_0 = 0.
    loss_1 = self.loss_1(outputs_1, labels_t_1)
    # loss = (loss_0 + loss_1) / 2.
    loss = loss_0 + loss_1
    return torch.unsqueeze(loss,0), outputs_0, outputs_1


class MyModel(nn.Module):
  def __init__(self, model, loss, strategy, loss_0_nll, loss_0_ce, loss_1_ce, edge_matrix_path="./edge_matrix_116.pt"):
    super(MyModel, self).__init__()
    self.use_two_head = loss == 'joint' # joint, lpl
    self.use_ssl = strategy != 'none' # strategy can be none, naive, filtering, conditioning
    
    self.model = model
    self.loss_0_nll = loss_0_nll 
    self.loss_0_ce = loss_0_ce 
    self.loss_1_ce = loss_1_ce 
    self.edge_matrix = torch.load(edge_matrix_path).float().cuda()
    self.log_softmax = torch.nn.LogSoftmax(dim=1)
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, inputs_t_0, inputs_t_1,
              labels_t_0_selftrain,
              labels_t_0_v1,
              labels_t_1_v1,
              labels_t_1_v2):
    # First compute labeled loss
    if self.use_two_head:
      outputs_t_1_v1, outputs_t_1_v2 = self.model(inputs_t_1)
    #   outputs_t_0_v1, outputs_t_0_v2 = self.model(inputs_t_0, return_both=self.use_ssl)
      outputs_t_0_v1, outputs_t_0_v2 = self.model(inputs_t_0, return_both=False)
    else:
      outputs_t_1_v2 = self.model(inputs_t_1)
      outputs_t_1_v1 = outputs_t_1_v2
      outputs_t_0_v1 = self.model(inputs_t_0)
      outputs_t_0_v2 = outputs_t_0_v1

    labeled_loss = self.loss_1_ce(outputs_t_1_v2, labels_t_1_v2)
    
    outputs_v1 = [torch.cat([outputs_t_0_v1[i], outputs_t_1_v1[i]]) for i in range(len(outputs_t_0_v1))]
    # for i in range(len(outputs_t_0_v1)):
    #   print(f"Shape is {outputs_v1[i].shape}")
    labels_v1 = torch.cat([labels_t_0_v1, labels_t_1_v1])
    if self.use_two_head:
      coarse_loss = self.loss_0_ce(outputs_v1, labels_v1)
    else:
      for i in range(len(outputs_v1)):
        outputs_v1[i] = outputs_v1[i] - outputs_v1[i].max(1)[0].unsqueeze(1)
        outputs_v1[i] = self.softmax(outputs_v1[i])
        outputs_v1[i] = torch.log(torch.matmul(outputs_v1[i].transpose(1,3), self.edge_matrix).transpose(1,3)  + 1e-20)
      coarse_loss = self.loss_0_nll(outputs_v1, labels_v1)
    #   coarse_loss = 0.
    
    if self.use_ssl:
      if self.use_two_head:
        loss = labeled_loss + coarse_loss
        loss.backward()
        labels_t_0_v1.cpu()
        labels_t_1_v1.cpu()
        labels_t_1_v2.cpu()
        inputs_t_0.cpu()
        inputs_t_1.cpu()
        del outputs_t_0_v1
        del outputs_t_1_v1
        del outputs_v1
        del outputs_t_1_v2
        torch.cuda.empty_cache()
        _, outputs_t_0_v2 = self.model(inputs_t_0, return_both=True)
        ssl_loss = self.loss_1_ce(outputs_t_0_v2, labels_t_0_selftrain)
        return torch.unsqueeze(ssl_loss,0)
      else:
        ssl_loss = self.loss_1_ce(outputs_t_0_v2, labels_t_0_selftrain)
    #   for i in range(len(outputs_v1)):
    #     outputs_t_0_v1[i].cpu()
    #     outputs_t_1_v1[i].cpu()
    #     outputs_v1[i].cpu()
    else:
      ssl_loss = 0.
    loss = labeled_loss + coarse_loss + ssl_loss
    return torch.unsqueeze(loss,0)


class MyModelNew(nn.Module):
  def __init__(self, model, loss, strategy, loss_0_nll, loss_0_ce, loss_1_ce, edge_matrix_path="./edge_matrix_116.pt"):
    super(MyModelNew, self).__init__()
    self.use_two_head = loss == 'joint' # joint, lpl
    self.use_ssl = strategy != 'none' # strategy can be none, naive, filtering, conditioning
    
    self.model = model
    self.loss_0_nll = loss_0_nll 
    self.loss_0_ce = loss_0_ce 
    self.loss_1_ce = loss_1_ce 
    self.edge_matrix = torch.load(edge_matrix_path).float().cuda()
    self.log_softmax = torch.nn.LogSoftmax(dim=1)
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, inputs_t_0, inputs_t_1,
              labels_t_0_selftrain,
              labels_t_0_v1,
              labels_t_1_v1,
              labels_t_1_v2):
    # First compute labeled loss
    if self.use_two_head:
      outputs_t_1_v2 = self.model(inputs_t_1, time=1)
    #   outputs_t_0_v1, outputs_t_0_v2 = self.model(inputs_t_0, return_both=self.use_ssl)
      outputs_t_0_v1 = self.model(inputs_t_0, time=0)
    else:
      outputs_t_1_v2 = self.model(inputs_t_1)
      outputs_t_0_v1 = self.model(inputs_t_0)
      outputs_t_0_v2 = [outputs_t_0_v1[i].clone() for i in range(len(outputs_t_0_v1))]
    
    labeled_loss = self.loss_1_ce(outputs_t_1_v2, labels_t_1_v2)
    
    if self.use_two_head:
      coarse_loss = self.loss_0_ce(outputs_t_0_v1, labels_t_0_v1)
    else:
      for i in range(len(outputs_t_0_v1)):
        outputs_t_0_v1[i] = outputs_t_0_v1[i] - outputs_t_0_v1[i].max(1)[0].unsqueeze(1)
        outputs_t_0_v1[i] = self.softmax(outputs_t_0_v1[i])
        outputs_t_0_v1[i] = torch.log(torch.matmul(outputs_t_0_v1[i].transpose(1,3), self.edge_matrix).transpose(1,3)  + 1e-20)
      coarse_loss = self.loss_0_nll(outputs_t_0_v1, labels_t_0_v1)
    #   coarse_loss = 0.
    
    if self.use_ssl:
      if self.use_two_head:
        # loss = labeled_loss + coarse_loss
        # loss.backward()
        # labels_t_0_v1.cpu()
        # labels_t_1_v1.cpu()
        # labels_t_1_v2.cpu()
        # inputs_t_0.cpu()
        # inputs_t_1.cpu()
        # del outputs_t_0_v1
        # del outputs_t_1_v1
        # del outputs_v1
        # del outputs_t_1_v2
        # torch.cuda.empty_cache()
        outputs_t_0_v2 = self.model(inputs_t_0, time=1)
        ssl_loss = self.loss_1_ce(outputs_t_0_v2, labels_t_0_selftrain)
        # return torch.unsqueeze(ssl_loss,0)
      else:
        ssl_loss = self.loss_1_ce(outputs_t_0_v2, labels_t_0_selftrain)
    #   for i in range(len(outputs_v1)):
    #     outputs_t_0_v1[i].cpu()
    #     outputs_t_1_v1[i].cpu()
    #     outputs_v1[i].cpu()
    else:
      ssl_loss = 0.
    loss = labeled_loss + coarse_loss + ssl_loss
    return torch.unsqueeze(loss,0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(exist_ok=True)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr