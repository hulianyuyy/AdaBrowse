import torch
weights_path = '/home/hulianyu/VAC_CSLR-main/work_dir/baseline_res18_image_scale_0_5/_best_model.pt'
state_dict = torch.load(weights_path)['model_state_dict']
#print(state_dict.keys())
for tmp in list(state_dict.keys()):
    print(tmp)
    if 'conv1d' in tmp or 'temporal_model' in tmp:
        del state_dict[tmp]
print('********************')
print(state_dict.keys())