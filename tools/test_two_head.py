import _init_paths
import models
import argparse
from config import config
from config import update_config
import torch

if __name__ == '__main__':
    # device = torch.device('cuda:{}'.format(0))    
    # torch.cuda.set_device(device)
    # torch.distributed.init_process_group(
    #     backend="nccl", init_method="env://",
    # )
    parser = argparse.ArgumentParser(description='Train segmentation network')
    args = parser.parse_args()
    args.cfg = "experiments/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_01.yaml"
    args.opts = []
    update_config(config, args)
    with torch.no_grad():
        model = models.seg_hrnet_ocr.get_seg_model(config)
        img = torch.ones((4,3,1000, 800))
        output = model(img)
        import pdb; pdb.set_trace()