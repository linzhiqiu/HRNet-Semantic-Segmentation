# on 2090
python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_2090_all_lr_0001.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_2090_all_lr_0001/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_2090_all.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_2090_all/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v1_2/v1_2_2090_half_0.yaml TEST.MODEL_FILE output/vista_v1_2/v1_2_2090_half_0/final_state.pth 

python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_all_lr_0001.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_all_lr_0001/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_all.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_all/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_half_0.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_half_0/final_state.pth 
python tools/test_vista.py --cfg experiments/vista_v2_0/v2_0_2090_half_1.yaml TEST.MODEL_FILE output/vista_v2_0/v2_0_2090_half_1/final_state.pth 