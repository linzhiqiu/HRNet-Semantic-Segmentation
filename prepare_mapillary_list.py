import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Mapillary Lists')
    
    parser.add_argument('--download_dir',
                        help='The directory with V2.0 dataset',
                        default='/data3/zhiqiul/mapillary_vistas',
                        type=str)
    parser.add_argument('--list_dir',
                        help='Generate list files under this directory',
                        default='./data/list')

    args = parser.parse_args()
    return args

def write_lst(file_name, lst):
    with open(file_name, 'w+') as file:
        for line in lst:
            file.write(line + "\n")

def main():
    args = parse_args()

    config_v1_2_path = args.download_dir / Path('config_{}.json'.format("v1.2"))
    config_v2_0_path = args.download_dir / Path('config_{}.json'.format("v2.0"))
    # read in config file
    with open(config_v1_2_path) as config_file:
        config_v1_2 = json.load(config_file)
    with open(config_v2_0_path) as config_file:
        config_v2_0 = json.load(config_file)

    # in this example we are only interested in the labels
    labels_v1_2 = config_v1_2['labels']
    labels_v2_0 = config_v2_0['labels']

    # print labels
    print("There are {} labels in the config file V1.2".format(len(labels_v1_2)))
    print("There are {} labels in the config file V2.0".format(len(labels_v2_0)))
    
    list_dir = Path(args.list_dir)
    list_dir.mkdir(exist_ok=True)
    
    list_dir_v1_2 = list_dir / "vista_v1_2"
    list_dir_v2_0 = list_dir / "vista_v2_0"
    list_dir_v1_2.mkdir(exist_ok=True, parents=True)
    list_dir_v2_0.mkdir(exist_ok=True, parents=True)

    list_dir_v1_2_train_all = list_dir_v1_2 / 'train.lst'
    list_dir_v1_2_train_half_0 = list_dir_v1_2 / 'train_half_0.lst'
    list_dir_v1_2_train_half_1 = list_dir_v1_2 / 'train_half_1.lst'
    list_dir_v1_2_val_all = list_dir_v1_2 / 'val.lst'
    list_dir_v1_2_val_half_0 = list_dir_v1_2 / 'val_half_0.lst'
    list_dir_v1_2_val_half_1 = list_dir_v1_2 / 'val_half_1.lst'
    list_dir_v1_2_test_all = list_dir_v1_2 / 'test.lst'

    list_dir_v2_0_train_all = list_dir_v2_0 / 'train.lst'
    list_dir_v2_0_train_half_0 = list_dir_v2_0 / 'train_half_0.lst'
    list_dir_v2_0_train_half_1 = list_dir_v2_0 / 'train_half_1.lst'
    list_dir_v2_0_val_all = list_dir_v2_0 / 'val.lst'
    list_dir_v2_0_val_half_0 = list_dir_v2_0 / 'val_half_0.lst'
    list_dir_v2_0_val_half_1 = list_dir_v2_0 / 'val_half_1.lst'
    list_dir_v2_0_test_all = list_dir_v2_0 / 'test.lst'
    
    all_ids = [p.stem for p in list((args.download_dir / Path("training/images")).glob("*.jpg"))]
    all_test_ids = [p.stem for p in list((args.download_dir / Path("validation/images")).glob("*.jpg"))]
    # import pdb; pdb.set_trace()
    v1_2_list = []
    v2_0_list = []
    for i in tqdm(all_ids):
        # image_path = Path(args.download_dir) / "training/images/{}.jpg".format(i)
        # label_path_v1_2 = Path(args.download_dir) / "training/{}/labels/{}.png".format('v1.2', i)
        # label_path_v2_0 = Path(args.download_dir) / "training/{}/labels/{}.png".format('v2.0', i)
        # assert image_path.exists()
        # assert label_path_v1_2.exists()
        # assert label_path_v2_0.exists()

        image_path = "training/images/{}.jpg".format(i)
        label_path_v1_2 = "training/{}/labels/{}.png".format('v1.2', i)
        label_path_v2_0 = "training/{}/labels/{}.png".format('v2.0', i)
        v1_2_list.append(f"{image_path} {label_path_v1_2}")
        v2_0_list.append(f"{image_path} {label_path_v2_0}")
    v1_2_list_half_0, v1_2_list_half_1 = v1_2_list[:9000], v1_2_list[9000:]
    v1_2_list_half_0_train, v1_2_list_half_0_val = v1_2_list_half_0[:8800], v1_2_list_half_0[8800:]
    v1_2_list_half_1_train, v1_2_list_half_1_val = v1_2_list_half_1[:8800], v1_2_list_half_1[8800:]
    v2_0_list_half_0, v2_0_list_half_1 = v2_0_list[:9000], v2_0_list[9000:]
    v2_0_list_half_0_train, v2_0_list_half_0_val = v2_0_list_half_0[:8800], v2_0_list_half_0[8800:]
    v2_0_list_half_1_train, v2_0_list_half_1_val = v2_0_list_half_1[:8800], v2_0_list_half_1[8800:]
    
    v1_2_list_train = v1_2_list_half_0_train + v1_2_list_half_1_train
    v2_0_list_train = v2_0_list_half_0_train + v2_0_list_half_1_train
    v1_2_list_val = v1_2_list_half_0_val + v1_2_list_half_1_val
    v2_0_list_val = v2_0_list_half_0_val + v2_0_list_half_1_val
    print(f"V1.2 Train size {len(v1_2_list_train)}; V1.2 Val size {len(v1_2_list_val)}")
    print(f"V2.0 Train size {len(v2_0_list_train)}; V2.0 Val size {len(v2_0_list_val)}")
    
    print(f"V1.2 Half 0 Train size {len(v1_2_list_half_0_train)}; V1.2 Val Half 0 size {len(v1_2_list_half_0_val)}")
    print(f"V2.0 Half 0 Train size {len(v2_0_list_half_0_train)}; V2.0 Half 0 Val size {len(v2_0_list_half_0_val)}")
    print(f"V1.2 Half 1 Train size {len(v1_2_list_half_1_train)}; V1.2 Val Half 1 size {len(v1_2_list_half_1_val)}")
    print(f"V2.0 Half 1 Train size {len(v2_0_list_half_1_train)}; V2.0 Half 1 Val size {len(v2_0_list_half_1_val)}")
    
    v1_2_list_test = []
    v2_0_list_test = []
    for i in tqdm(all_test_ids):
        image_path = "validation/images/{}.jpg".format(i)
        label_path_v1_2 = "validation/{}/labels/{}.png".format('v1.2', i)
        label_path_v2_0 = "validation/{}/labels/{}.png".format('v2.0', i)
        v1_2_list_test.append(f"{image_path} {label_path_v1_2}")
        v2_0_list_test.append(f"{image_path} {label_path_v2_0}")
    print(f"V1.2 TEST size {len(v1_2_list_test)}")
    print(f"V2.0 TEST size {len(v2_0_list_test)}")

    write_lst(list_dir_v1_2_train_all, v1_2_list_train)
    write_lst(list_dir_v1_2_train_half_0, v1_2_list_half_0_train)
    write_lst(list_dir_v1_2_train_half_1, v1_2_list_half_1_train)
    write_lst(list_dir_v1_2_val_all, v1_2_list_val)
    write_lst(list_dir_v1_2_val_half_0, v1_2_list_half_0_val)
    write_lst(list_dir_v1_2_val_half_1, v1_2_list_half_1_val)
    write_lst(list_dir_v1_2_test_all, v1_2_list_test)

    write_lst(list_dir_v2_0_train_all, v2_0_list_train)
    write_lst(list_dir_v2_0_train_half_0, v2_0_list_half_0_train)
    write_lst(list_dir_v2_0_train_half_1, v2_0_list_half_1_train)
    write_lst(list_dir_v2_0_val_all, v2_0_list_val)
    write_lst(list_dir_v2_0_val_half_0, v2_0_list_half_0_val)
    write_lst(list_dir_v2_0_val_half_1, v2_0_list_half_1_val)
    write_lst(list_dir_v2_0_test_all, v2_0_list_test)

if __name__ == "__main__":
    main()
