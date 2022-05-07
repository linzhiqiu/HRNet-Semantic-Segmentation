import torch
modes = [
    'finetune/finetune_on_upper',
    'finetune/finetune_on_half_0',
    'finetune/finetune_on_half_1',
    'all',
    'half_0',
    'finetune/single_head',
    'finetune/two_head',
]
for mode in modes:
    print()
    print(f"Mode : {mode}")
    result_file = "output/vista_v2_0/{}/test_result.pt"
    print(torch.load(result_file))
