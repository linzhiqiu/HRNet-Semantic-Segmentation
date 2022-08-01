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
(autobot-0-25) CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_1/final_state.pth

(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head/final_state.pth
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_finetune_on_half_0/final_state.pth
(trinity-2-9-vnice) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head/final_state.pth
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
() torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_two_head.py --cfg experiments/vista_v1_2_10000/finetune.yaml 





# self-training (TODO: Rerun before must rsync)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --samples 2500 --strategy none
CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --samples 2500 --strategy filtering
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --samples 2500 --strategy conditioning
CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_10000/finetune.yaml --samples 5000 --strategy none
CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_10000/finetune.yaml --samples 5000 --strategy filtering
CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_10000/finetune.yaml --samples 5000 --strategy conditioning

(rsync first?) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --strategy none
(rsync first?) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --strategy conditioning
(rsync first?) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --strategy filtering
(rsync first?) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_10000.py --cfg experiments/vista_v1_2_10000/finetune.yaml --strategy conditioning
(rsync first?) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_10000.py --cfg experiments/vista_v1_2_10000/finetune.yaml --strategy none
(rsync first?) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_10000.py --cfg experiments/vista_v1_2_10000/finetune.yaml --strategy filtering

(running) python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning/final_state.pth
CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none/final_state.pth
CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering/final_state.pth

  # sanity checking the pseudolabel quality + fixing the labels
  python tools/self_train_test.py --cfg experiments/vista_v1_2_10000/finetune.yaml --samples 5000 --strategy none
  python tools/self_train_test.py --cfg experiments/vista_v1_2_10000/finetune.yaml --samples 5000 --strategy filtering
  python tools/self_train_test.py --cfg experiments/vista_v1_2_10000/finetune.yaml --samples 5000 --strategy conditioning
  python tools/self_train_test.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --samples 2500 --strategy none
  python tools/self_train_test.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --samples 2500 --strategy filtering
  python tools/self_train_test.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml --samples 2500 --strategy conditioning
  python tools/self_train_test.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --samples 9000 --strategy none
  python tools/self_train_test.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --samples 9000 --strategy filtering
  python tools/self_train_test.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --samples 9000 --strategy conditioning

# consistent testing
(0.3021) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head_consistent/final_state.pth
(0.2932) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head_consistent/epoch_699.pth

# 2500 no imagenet
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v1_2_5000/2500_half_0_scratch.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_5000/2500_half_1_scratch.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_5000/2500_all_scratch.yaml 

python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_half_0_scratch/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_half_1_scratch/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_all_scratch/final_state.pth

# finetune from scratch 5000 samples
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_scratch_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml --mode upper
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_scratch_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml --mode half_0
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_scratch_vista_5000.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml --mode half_1
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_single_head_scratch.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_two_head_scratch.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml 


(running) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_upper/final_state.pth
(running) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_half_1/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_single_head_scratch/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_half_0/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune_scratch.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_two_head_scratch/final_state.pth


(running) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch_mini.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_upper/final_state.pth
(running) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch_mini.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_half_1/final_state.pth
(running) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_upper/final_state.pth
(running) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_scratch_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune_scratch/2500_finetune_scratch_on_half_1/final_state.pth


# double weight + consistent
(trinity-2-1) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_single_head_double_weight.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
() torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_two_head_double_weight.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 

python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head_double_weight/final_state.pth
CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head_double_weight/final_state.pth


# combining all and fix single/two head loss (include label set + remove average)
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy none --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy naive --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy filtering --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy conditioning --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy none --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy naive --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy filtering --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy conditioning --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 

(MeanIU:  0.2948, Pixel_Acc:  0.8942,         Mean_Acc:  0.3743)python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_lpl_corrected_double_False/final_state.pth
(MeanIU:  0.2928, Pixel_Acc:  0.8952,         Mean_Acc:  0.3719)python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_lpl_corrected_double_False/final_state.pth
(MeanIU:  0.2860, Pixel_Acc:  0.8954,         Mean_Acc:  0.3564)python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_lpl_corrected_double_False/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_lpl_corrected_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_joint_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_joint_double_False/final_state.pth
(MeanIU:  0.2397, Pixel_Acc:  0.8834,         Mean_Acc:  0.3078) python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_joint_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_joint_double_False/final_state.pth

# minimal test
CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_lpl_corrected_double_False/final_state.pth
(MeanIU:  0.2808, Pixel_Acc:  0.8903, Mean_Acc:  0.3586) CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_lpl_corrected_double_False/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_lpl_corrected_double_False/final_state.pth
python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_lpl_corrected_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_joint_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_joint_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_joint_double_False/final_state.pth
python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_joint_double_False/final_state.pth

# testing for no avg
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_two_head.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(MeanIU:  0.3106, Pixel_Acc:  0.8991,  Mean_Acc:  0.3912) python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head_consistent_no_avg/final_state.pth
 # minimal: (MeanIU:  0.3033, Pixel_Acc:  0.8945, Mean_Acc:  0.3831) CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune_minimal.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_two_head_consistent_no_avg/final_state.pth


# corrected the lpl loss (should have no difference)
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final.py --strategy conditioning --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml

# test t0 performance..
MeanIU:  0.3701, Pixel_Acc:  0.8927,         Mean_Acc:  0.4479 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_half_0.yaml TEST.MODEL_FILE output/vista_v1_2/2500_half_0/final_state.pth

MeanIU:  0.2822, Pixel_Acc:  0.8916,         Mean_Acc:  0.3499 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_half_0/final_state.pth
MeanIU:  0.2928, Pixel_Acc:  0.8934,         Mean_Acc:  0.3695 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_half_1/final_state.pth
MeanIU:  0.3071, Pixel_Acc:  0.8994,         Mean_Acc:  0.3859, python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_all/final_state.pth

# testing for no avg
torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista_single_head.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(MeanIU:  0.3099, Pixel_Acc:  0.8987, Mean_Acc:  0.3855) python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_single_head_consistent_no_avg/final_state.pth

# combining all and fix single/two head loss (include label set + remove average)
(finished) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy none --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy naive --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy filtering --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy conditioning --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(finished) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy none --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(finished) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy naive --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy filtering --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(finished) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy conditioning --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 

MeanIU:  0.3105, Pixel_Acc:  0.8996,         Mean_Acc:  0.3893 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_joint_double_False_new/final_state.pth
MeanIU:  0.2702, Pixel_Acc:  0.8893,         Mean_Acc:  0.3288 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_joint_double_False_new/final_state.pth
CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_joint_double_False_new/final_state.pth
MeanIU:  0.2778, Pixel_Acc:  0.8903,         Mean_Acc:  0.3467 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_joint_double_False_new/final_state.pth
MeanIU:  0.3104, Pixel_Acc:  0.8983,         Mean_Acc:  0.3900 CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_lpl_double_False_new/final_state.pth
MeanIU:  0.3165, Pixel_Acc:  0.8997,         Mean_Acc:  0.3901 CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_lpl_double_False_new/final_state.pth
MeanIU:  0.3141, Pixel_Acc:  0.8991,         Mean_Acc:  0.3956 CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_lpl_double_False_new/final_state.pth
MeanIU:  0.3123, Pixel_Acc:  0.8983,         Mean_Acc:  0.3927 CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_lpl_double_False_new/final_state.pth


# correct grad
(running-trinity) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy naive --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy filtering --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy conditioning --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 

MeanIU:  0.3109, Pixel_Acc:  0.8970,         Mean_Acc:  0.3842, CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_naive_loss_joint_double_False_correctgrad/final_state.pth
MeanIU:  0.3157, Pixel_Acc:  0.8994,         Mean_Acc:  0.3988, CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_filtering_loss_joint_double_False_correctgrad/final_state.pth
MeanIU:  0.3150, Pixel_Acc:  0.8988,         Mean_Acc:  0.3976 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_conditioning_loss_joint_double_False_correctgrad/final_state.pth


# 18000
(running-trinity) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v1_2_18000/9000_half_0.yaml 
(running) /project_data/ramanan/mengtial/scripts/vnice/vnice.sh torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_18000/9000_half_1.yaml 
(running) /project_data/ramanan/mengtial/scripts/vnice/vnice.sh torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_18000/9000_half_0.yaml 
(running) /project_data/ramanan/mengtial/scripts/vnice/vnice.sh torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/train_vista.py --cfg experiments/vista_v2_0_18000/9000_all.yaml 

#t0 performance
'mean_IoU': 0.44139174714573576, 'pixel_acc': 0.9066033107288457, 'mean_acc': 0.5424515076849, CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_half_0.yaml TEST.MODEL_FILE output/vista_v1_2/9000_half_0/final_state.pth

# t1 performance
'mean_IoU': 0.328312594952233 'pixel_acc': 0.9053123580544007, 'mean_acc': 0.4067558122386495, MeanIU:  0.3283, Pixel_Acc:  0.9053,         Mean_Acc:  0.4068 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_half_1/final_state.pth
MeanIU:  0.3344, Pixel_Acc:  0.9044,         Mean_Acc:  0.4196 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_half_0/final_state.pth
'mean_IoU': 0.3382167656536317,'pixel_acc': 0.9088853543296879, 'mean_acc': 0.42582476650901113, CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_all/final_state.pth

# finetunePrev
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --mode upper
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --mode half_0
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --mode half_1

# test 
(MeanIU:  0.3620, Pixel_Acc:  0.9099,         Mean_Acc:  0.4472) CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_finetune_on_upper/final_state.pth
(MeanIU:  0.3511, Pixel_Acc:  0.9050,         Mean_Acc:  0.4343) CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_finetune_on_half_0/final_state.pth
(MeanIU:  0.3397, Pixel_Acc:  0.9052,         Mean_Acc:  0.4169) CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_finetune_on_half_1/final_state.pth

# self-train prepare
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --samples 9000 --strategy none
CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --samples 9000 --strategy filtering
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/self_train.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --samples 9000 --strategy conditioning


# combining all and 18000
(done) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy none --samples 9000 --loss joint --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(running autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy naive --samples 9000 --loss joint --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(running autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy filtering --samples 9000 --loss joint --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(running trinity) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy conditioning --samples 9000 --loss joint --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(done) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy none --samples 9000 --loss lpl --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(running autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy naive --samples 9000 --loss lpl --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(running trinity) /data3/shared/scripts/vnice/vnice.sh torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy filtering --samples 9000 --loss lpl --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(done) /data3/shared/scripts/vnice/vnice.sh torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new.py --strategy conditioning --samples 9000 --loss lpl --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 

MeanIU:  0.3504, Pixel_Acc:  0.9077,         Mean_Acc:  0.4444, CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_none_loss_lpl_double_False_correctgrad/final_state.pth
MeanIU:  0.3597, Pixel_Acc:  0.9099,         Mean_Acc:  0.4467 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_naive_loss_lpl_double_False_correctgrad/final_state.pth
MeanIU:  0.3612, Pixel_Acc:  0.9103,         Mean_Acc:  0.4530,CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_filtering_loss_lpl_double_False_correctgrad/final_state.pth
MeanIU:  0.3547, Pixel_Acc:  0.9088,         Mean_Acc:  0.4485, CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_conditioning_loss_lpl_double_False_correctgrad/final_state.pth
MeanIU:  0.3540, Pixel_Acc:  0.9096,         Mean_Acc:  0.4469, CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_none_loss_joint_double_False_correctgrad/final_state.pth
MeanIU:  0.3572, Pixel_Acc:  0.9081,         Mean_Acc:  0.4402 CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_naive_loss_joint_double_False_correctgrad/final_state.pth
MeanIU:  0.3606, Pixel_Acc:  0.9114,         Mean_Acc:  0.4465, CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_filtering_loss_joint_double_False_correctgrad/final_state.pth
MeanIU:  0.3570, Pixel_Acc:  0.9091,         Mean_Acc:  0.4512CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_conditioning_loss_joint_double_False_correctgrad/final_state.pth

# ssl only 18000
(running autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --strategy conditioning
(running trinity-2-5) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --strategy none
(running autobot) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_selftrain_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --strategy filtering

# wrong result with only pseudo-label
#MeanIU:  0.3239, Pixel_Acc:  0.9034,         Mean_Acc:  0.3944, CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_none/final_state.pth
#MeanIU:  0.3384, Pixel_Acc:  0.9085,         Mean_Acc:  0.4103, CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_filtering/final_state.pth
#MeanIU:  0.3361, Pixel_Acc:  0.9042,         Mean_Acc:  0.4180, Class IoU: python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_conditioning/final_state.pth

# correct result
MeanIU:  0.3268, Pixel_Acc:  0.9036,         Mean_Acc:  0.3982 CUDA_VISIBLE_DEVICES=6,1,2,3,4,5,0,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_none/final_state.pth
MeanIU:  0.3496, Pixel_Acc:  0.9108,         Mean_Acc:  0.4255,CUDA_VISIBLE_DEVICES=7,1,2,3,4,5,6,0 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_filtering/final_state.pth
MeanIU:  0.3529, Pixel_Acc:  0.9097,         Mean_Acc:  0.4360 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_conditioning/final_state.pth


# finetunePrev second run
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --mode upper
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --mode half_0
(running) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_18000.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml --mode half_1

# test 
CUDA_VISIBLE_DEVICES=3,1,2,0,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_finetune_on_upper_second_run/final_state.pth
MeanIU:  0.3501, Pixel_Acc:  0.9050,         Mean_Acc:  0.4329, CUDA_VISIBLE_DEVICES=4,1,2,3,0,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_finetune_on_half_0_second_run/final_state.pth
MeanIU:  0.3408, Pixel_Acc:  0.9052,         Mean_Acc:  0.4184, CUDA_VISIBLE_DEVICES=5,1,2,3,4,0,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_finetune_on_half_1_second_run/final_state.pth




###### New Upper Bound
# combining all and 18000
(done) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new_upperbound.py --strategy none --samples 9000 --loss joint --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 
(done) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new_upperbound.py --strategy none --samples 9000 --loss lpl --cfg experiments/vista_v1_2_18000/9000_finetune.yaml 

CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_none_loss_lpl_double_False_newupper/final_state.pth
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_18000/9000_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/9000_finetune/9000_selftrain_none_loss_joint_double_False_newupper/final_state.pth

# combining all and 5000
(done) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new_upperbound.py --strategy none --samples 2500 --loss joint --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 
(done) torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/finetune_vista_final_new_upperbound.py --strategy none --samples 2500 --loss lpl --cfg experiments/vista_v1_2_5000/2500_finetune.yaml 

CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 python tools/test_vista.py --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_lpl_double_False_newupper/final_state.pth
CUDA_VISIBLE_DEVICES=2,1,0,3,4,5,6,7 python tools/test_vista.py --mode two_head --cfg experiments/vista_v1_2_5000/2500_finetune.yaml TEST.MODEL_FILE output/vista_v2_0/2500_finetune/2500_selftrain_none_loss_joint_double_False_newupper/final_state.pth
