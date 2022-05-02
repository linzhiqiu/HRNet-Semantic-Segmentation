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
                        default='/project_data/ramanan/zhiqiu/mapillary_vistas',
                        type=str)
    parser.add_argument('--list_dir',
                        help='Generate list files under this directory',
                        default='./data/list')
    parser.add_argument('--num_samples',
                        help='Num of samples per timestamp',
                        type=int,
                        default=None)

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
        
    num_samples = args.num_samples

    # in this example we are only interested in the labels
    labels_v1_2 = config_v1_2['labels']
    labels_v2_0 = config_v2_0['labels']

    # print labels
    print("There are {} labels in the config file V1.2".format(len(labels_v1_2)))
    print("There are {} labels in the config file V2.0".format(len(labels_v2_0)))
    
    list_dir = Path(args.list_dir)
    list_dir.mkdir(exist_ok=True)
    
    if num_samples:
        num_samples_str = f"_{num_samples*2}"
    else:
        num_samples_str = ""
    list_dir_v1_2 = list_dir / f"vista_v1_2{num_samples_str}"
    list_dir_v2_0 = list_dir / f"vista_v2_0{num_samples_str}"
    list_dir_v1_2.mkdir(exist_ok=True, parents=True)
    list_dir_v2_0.mkdir(exist_ok=True, parents=True)

    list_dir_v1_2_train_all = list_dir_v1_2 / 'train.lst'
    list_dir_v1_2_train_half_0 = list_dir_v1_2 / 'train_half_0.lst'
    list_dir_v1_2_train_half_1 = list_dir_v1_2 / 'train_half_1.lst'
    list_dir_v1_2_test_all = list_dir_v1_2 / 'test.lst'

    list_dir_v2_0_train_all = list_dir_v2_0 / 'train.lst'
    list_dir_v2_0_train_half_0 = list_dir_v2_0 / 'train_half_0.lst'
    list_dir_v2_0_train_half_1 = list_dir_v2_0 / 'train_half_1.lst'
    list_dir_v2_0_test_all = list_dir_v2_0 / 'test.lst'
    
    all_ids = [p.stem for p in list((args.download_dir / Path("training/images")).glob("*.jpg"))]
    all_test_ids = [p.stem for p in list((args.download_dir / Path("validation/images")).glob("*.jpg"))]
    # import pdb; pdb.set_trace()
    v1_2_list = []
    v2_0_list = []
    for i in tqdm(all_ids):
        image_path = "training/images/{}.jpg".format(i)
        label_path_v1_2 = "training/{}/labels/{}.png".format('v1.2', i)
        label_path_v2_0 = "training/{}/labels/{}.png".format('v2.0', i)
        v1_2_list.append(f"{image_path} {label_path_v1_2}")
        v2_0_list.append(f"{image_path} {label_path_v2_0}")
    if num_samples:
        v1_2_list = v1_2_list[:num_samples*2]
        v2_0_list = v2_0_list[:num_samples*2]
    length = int(len(v1_2_list) / 2)
    v1_2_list_half_0, v1_2_list_half_1 = v1_2_list[:length], v1_2_list[length:]
    v2_0_list_half_0, v2_0_list_half_1 = v2_0_list[:length], v2_0_list[length:]
    
    print(f"V1.2 Train size {len(v1_2_list_half_0)}; V1.2 Val size {len(v1_2_list_half_1)}")
    print(f"V2.0 Train size {len(v2_0_list_half_0)}; V2.0 Val size {len(v2_0_list_half_1)}")
    
    
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

    write_lst(list_dir_v1_2_train_all, v1_2_list)
    write_lst(list_dir_v1_2_train_half_0, v1_2_list_half_0)
    write_lst(list_dir_v1_2_train_half_1, v1_2_list_half_1)
    write_lst(list_dir_v1_2_test_all, v1_2_list_test)

    write_lst(list_dir_v2_0_train_all, v2_0_list)
    write_lst(list_dir_v2_0_train_half_0, v2_0_list_half_0)
    write_lst(list_dir_v2_0_train_half_1, v2_0_list_half_1)
    write_lst(list_dir_v2_0_test_all, v2_0_list_test)

if __name__ == "__main__":
    main()
