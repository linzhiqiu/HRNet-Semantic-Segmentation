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
(not evaling) CUDA_VISIBLE_DEVICES=2,0,1,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_150_epochs_720_size_lr_005.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_150_epochs_720_size_lr_005/final_state.pth
(not evaling) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0_300_epochs_720_size_lr_005.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0_300_epochs_720_size_lr_005/final_state.pth
(not evaling) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_01.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_100_epochs_720_size_lr_01/final_state.pth
(not running) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_half_0_200_epochs_720_size_lr_01.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_half_0_200_epochs_720_size_lr_01/final_state.pth
(not running) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_3090_all_150_epochs_720_size_lr_0003.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_3090_all_150_epochs_720_size_lr_0003/final_state.pth


# 10000 samples
(evaling) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/half_0.yaml TEST.MODEL_FILE output/vista_v1_2/half_0/final_state.pth
(evaling) CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v2_0_10000/half_0.yaml TEST.MODEL_FILE output/vista_v2_0/half_0/final_state.pth
(trinity to be transferred) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v2_0_10000/all.yaml TEST.MODEL_FILE output/vista_v2_0_10000/all/final_state.pth
(on trinity stopped) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/all.yaml TEST.MODEL_FILE output/vista_v1_2/all/final_state.pth

(running on autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_two_head.py --cfg experiments/vista_v1_2_10000/finetune.yaml
(running on autobot-vnice) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_single_head.py --cfg experiments/vista_v1_2_10000/finetune.yaml
(running on autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml --mode half_1
(running on trinity) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml --mode upper
(running on trinity-vnice) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml --mode half_0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_upper/final_state.pth
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_half_0/final_state.pth
CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_half_1/final_state.pth
CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/all/final_state.pth
CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/half_0/final_state.pth
CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/single_head/final_state.pth
CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/two_head/final_state.pth

# try different tau for 10000 samples
(autobot-0-25) CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.5 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_upper/final_state.pth 
(autobot-0-25) CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.5 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_half_1/final_state.pth 
(autobot-0-25) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista_tau.py --tau 1.5 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/single_head/final_state.pth 
(autobot-0-25) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista_tau.py --mode two_head --tau 1.5 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/two_head/final_state.pth 

(autobot-0-25) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista_tau.py --tau 1.9 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_upper/final_state.pth 
(autobot-0-25) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista_tau.py --tau 1.9 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_half_1/final_state.pth 
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.9 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/single_head/final_state.pth 
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista_tau.py --mode two_head --tau 1.9 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/two_head/final_state.pth 

(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.2 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_upper/final_state.pth 
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista_tau.py --tau 1.2 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/finetune_on_half_1/final_state.pth 
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista_tau.py --tau 1.2 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/single_head/final_state.pth 
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista_tau.py --mode two_head --tau 1.2 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/finetune/two_head/final_state.pth 

# 5000 samples
(running on trinity-vnice) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v1_2_5000/2500_half_0.yaml 
(autobot-0-33) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_5000/2500_half_0.yaml 
(autobot-0-29) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_5000/2500_all.yaml 
(autobot-0-25) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_5000/2500_half_1.yaml 
# finetune 5000 samples
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --mode upper
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --mode half_0
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --mode half_1
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_single_head.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_two_head.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 


(autobot-0-25) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_upper/final_state.pth
(autobot-0-25) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_1/final_state.pth

(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_0/final_state.pth
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head/final_state.pth
# CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/all/final_state.pth
# CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/half_0/final_state.pth

# tau normal on 5000 samples
(trinity-2-1) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.2 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_upper/final_state.pth
(trinity-2-1) CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.2 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_1/final_state.pth
(trinity-2-1) CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista_tau.py --tau 1.2 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
(trinity-2-1) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista_tau.py --mode two_head --tau 1.2 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head/final_state.pth

(trinity-2-1) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista_tau.py --tau 1.0 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
(trinity-2-1) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista_tau.py --tau 0.8 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
(trinity-2-1) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista_tau.py --tau 0.5 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
(trinity-2-1) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista_tau.py --tau 0.3 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth

(TODO) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista_tau.py --tau 1.5 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_upper/final_state.pth
(TODO) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista_tau.py --tau 1.5 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_1/final_state.pth
(TODO) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista_tau.py --tau 1.5 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
(TODO) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista_tau.py --mode two_head --tau 1.5 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head/final_state.pth

(TODO) python tools/test_vista_tau.py --tau 1.9 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_upper/final_state.pth
(TODO) python tools/test_vista_tau.py --tau 1.9 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_1/final_state.pth
(TODO) python tools/test_vista_tau.py --tau 1.9 --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
(TODO) python tools/test_vista_tau.py --mode two_head --tau 1.9 --cfg experiments/vista_v1_2_10000/finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head/final_state.pth

# 2500 consistent single head
(trinity-2-1) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_single_head.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(autobot-0-25) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_two_head.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
() torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_two_head.py --cfg experiments/vista_v1_2_5000/finetune.yaml 