# on 2090
python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_2090_all_lr_0001.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_2090_all_lr_0001/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_2090_all.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_2090_all/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_2090_half_0.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_2090_half_0/final_state.pth 

python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_all_lr_0001.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_all_lr_0001/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_all.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_all/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_half_0.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_half_0/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_half_1.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_half_1/final_state.pth 

# 3090
(0.38) CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_0003.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_0003/final_state.pth 
(0.4547) CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_100_epochs_800_size.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_100_epochs_800_size/final_state.pth 
(0.3780) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_100_epochs_800_size_lr_00003.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_100_epochs_800_size_lr_00003/final_state.pth
(0.3663) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0/final_state.pth
(0.3900) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0_200_epochs.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0_200_epochs/final_state.pth
(0.4505) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0_200_epochs_800_size.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0_200_epochs_800_size/final_state.pth 
(training) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_005.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_005/final_state.pth
(training) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0_200_epochs_720_size_lr_005.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0_200_epochs_720_size_lr_005/final_state.pth
(to be eval) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_01.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_01/final_state.pth
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0_200_epochs_720_size_lr_01.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0_200_epochs_720_size_lr_01/final_state.pth
(not running) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_150_epochs_720_size_lr_0003.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_150_epochs_720_size_lr_0003/final_state.pth