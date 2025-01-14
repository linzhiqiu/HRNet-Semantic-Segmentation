# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image
import json
import torch
import random
from tqdm import tqdm
from torch.nn import functional as F

from .base_dataset import BaseDataset

class Vista(BaseDataset):
    def __init__(self,
                 root,
                 list_path=[], 
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(1024, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 class_balancing=False,
                 eval_all=False, # whether or not to include void class(es)
                 no_label_map=False,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Vista, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)
        self.root = root # root dir for downloaded data
        self.list_path = list_path
        self.eval_all = eval_all
        self.multi_scale = multi_scale
        self.flip = flip
        self.no_label_map = no_label_map
        self.img_list = [line.strip().split() for line in open(list_path)]

        self.files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            self.files.append({
                "img": os.path.join(self.root, image_path),
                "label": os.path.join(self.root, label_path),
                "name": name,
            })
        
        self.config_path = os.path.join(self.root, 'config_{}.json'.format(self.version))
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)
        self.labels = self.config['labels']
        
        self.num_classes = 0
        self.label_mapping = {}
        for idx, l in enumerate(self.labels):
            if no_label_map:
                self.label_mapping[idx] = idx
            elif l['evaluate'] or self.eval_all:
                self.label_mapping[idx] = self.num_classes
                self.num_classes += 1
            else:
                self.label_mapping[idx] = ignore_label # 255
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843, 
        #                                 1.1116, 0.9037, 1.0865, 1.0955, 
        #                                 1.0865, 1.1529, 1.0507]).cuda()
        self.class_balancing = class_balancing
        if self.class_balancing:
            class_weights_path = os.path.join(self.root, f'class_weights_{self.version}.pt')
            if os.path.isfile(class_weights_path):
                print(f"Loading from {class_weights_path}")
                self.class_weights = torch.load(class_weights_path).cuda()
                print(f"Loaded from {class_weights_path}")
                print(self.class_weights)
            else:
                self.class_weights = torch.zeros(self.num_classes).cuda()
                for f in tqdm(self.files):
                    label = np.array(Image.open(f['label']))
                    # convert labeled data to numpy arrays for better handling
                    label = self.convert_label(label)
                    for label_id in np.unique(label):
                        if label_id == ignore_label:
                            continue
                        sum_id = (label == label_id).sum()
                        self.class_weights[label_id] += sum_id * 0.000001
                total_sum = self.class_weights.sum()
                self.class_weights = total_sum / self.class_weights # This is inverse class ratio
                self.class_weights = self.class_weights / self.class_weights.mean()
                print(f"Saving to {class_weights_path}")
                torch.save(self.class_weights, class_weights_path)
        else:
            self.class_weights = None
        print('Class weighting is:')
        print(self.class_weights)
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()  
        if self.no_label_map:
            return label
        elif inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"],
                           cv2.IMREAD_COLOR)
        size = image.shape

        label_path = item['label']
        label = np.array(Image.open(label_path))
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                            self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.cpu().numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width])
        for scale in scales:
            # print(f"Scale: {scale}")
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
            old_height, old_width = image.shape[:-1]
            # print(f"Scale is {scale}")
            # print(f"Height width is {old_height}->{height} {old_width}->{width}")
            if height < stride_h and width < stride_w:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                # preds = self.inference(config, model.module.model, new_img, flip)
                preds = self.inference(config, model.module, new_img, flip).cpu()
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = int(np.ceil(1.0 * (new_h - self.crop_size[0]) / stride_h)) + 1
                cols = int(np.ceil(1.0 * (new_w - self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes, new_h,new_w])
                count = torch.zeros([1,1, new_h, new_w])

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model.module, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0].cpu()
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

class Vista_V1_2(Vista):
    def __init__(self,
                 root,
                 list_path=[],
                 multi_scale=True, 
                 flip=True,
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(1024, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 class_balancing=False,
                 eval_all=True,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        self.version = 'v1.2'
        super(Vista_V1_2, self).__init__(root,
                                         list_path=list_path,
                                         multi_scale=multi_scale, 
                                         flip=flip, 
                                         ignore_label=ignore_label, 
                                         base_size=base_size, 
                                         crop_size=crop_size, 
                                         downsample_rate=downsample_rate,
                                         scale_factor=scale_factor,
                                         class_balancing=class_balancing,
                                         eval_all=eval_all,
                                         mean=mean, 
                                         std=std)

class Vista_V2_0(Vista):
    def __init__(self,
                 root,
                 list_path=[], 
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(1024, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 class_balancing=False,
                 eval_all=False,
                 no_label_map=False,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        self.version = 'v2.0'
        super(Vista_V2_0, self).__init__(root,
                                         list_path=list_path,
                                         multi_scale=multi_scale, 
                                         flip=flip, 
                                         ignore_label=ignore_label, 
                                         base_size=base_size, 
                                         crop_size=crop_size, 
                                         downsample_rate=downsample_rate,
                                         scale_factor=scale_factor,
                                         class_balancing=class_balancing,
                                         eval_all=eval_all,
                                         no_label_map=no_label_map,
                                         mean=mean, 
                                         std=std)
        
class VistaBoth(BaseDataset):
    def __init__(self,
                 root,
                 list_path_0=None,
                 list_path_1=None,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(1024, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 no_label_map=[False, False],
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(VistaBoth, self).__init__(ignore_label, base_size,
            crop_size, downsample_rate, scale_factor, mean, std)
        self.root = root # root dir for downloaded data
        self.list_path_0 = list_path_0
        self.list_path_1 = list_path_1
        self.no_label_map = no_label_map
        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list_0 = [line.strip().split() for line in open(list_path_0)]
        self.img_list_1 = [line.strip().split() for line in open(list_path_1)]

        self.files = []
        for item_0, item_1 in zip(self.img_list_0, self.img_list_1):
            image_path_0, label_path_0 = item_0
            image_path_1, label_path_1 = item_1
            assert image_path_0 == image_path_1
            name = os.path.splitext(os.path.basename(label_path_0))[0]
            self.files.append({
                "img": os.path.join(self.root, image_path_0),
                "labels": [os.path.join(self.root, label_path_0), os.path.join(self.root, label_path_1)],
                "name": name,
            })
        
        self.config_path = os.path.join(self.root, 'config_v2.0.json')
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)
        self.labels = self.config['labels']
        
        self.num_classes = 0
        self.label_mapping = {}
        for idx, l in enumerate(self.labels):
            if l['evaluate']:
                self.label_mapping[idx] = self.num_classes
                self.num_classes += 1
            else:
                self.label_mapping[idx] = ignore_label # 255
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()  
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    
    def rand_crop(self, image, label_0, label_1):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label_0 = self.pad_image(label_0, h, w, self.crop_size,
                               (self.ignore_label,))
        label_1 = self.pad_image(label_1, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label_0.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label_0 = label_0[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label_1 = label_1[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label_0, label_1
    
    def multi_scale_aug(self, image, label_0, label_1,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        label_0 = cv2.resize(label_0, (new_w, new_h),
                            interpolation=cv2.INTER_NEAREST)
        label_1 = cv2.resize(label_1, (new_w, new_h),
                            interpolation=cv2.INTER_NEAREST)

        if rand_crop:
            image, label_0, label_1 = self.rand_crop(image, label_0, label_1)

        return image, label_0, label_1

    
    def gen_sample(self, image, label_0, label_1,
                   multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label_0, label_1 = self.multi_scale_aug(image, label_0, label_1,
                                                rand_scale=rand_scale)

        # image = self.random_brightness(image)
        image = self.input_transform(image)
        label_0 = self.label_transform(label_0)
        label_1 = self.label_transform(label_1)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label_0 = label_0[:, ::flip]
            label_1 = label_1[:, ::flip]

        if self.downsample_rate != 1:
            label_0 = cv2.resize(
                label_0,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )
            label_1 = cv2.resize(
                label_1,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label_0, label_1

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"],
                           cv2.IMREAD_COLOR)
        size = image.shape

        label_path_0, label_path_1 = item['labels']
        label_0 = np.array(Image.open(label_path_0))
        label_1 = np.array(Image.open(label_path_1))
        
        if not self.no_label_map[0]:
            label_0 = self.convert_label(label_0)
        if not self.no_label_map[1]:
            label_1 = self.convert_label(label_1)

        image, label_0, label_1 = self.gen_sample(image, label_0, label_1, 
                            self.multi_scale, self.flip)

        return image.copy(), label_0.copy(), label_1.copy(), np.array(size), name