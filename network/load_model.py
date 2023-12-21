from configs.params import *
import torch
import configs.params as params

device = params.device

def load_pretrained_network(model_, path, device, extra_dict='', backbone_nm=''):
    model_.eval().to(device)
    
    state_dict_loaded = model_.state_dict()
    if extra_dict != '':
        state_dict_pretrained = torch.load(path, map_location=device)[extra_dict]
    else:
        state_dict_pretrained = torch.load(path, map_location=device)
    state_dict_temp = {}

    for k in state_dict_loaded:
        state_dict_temp[k] = state_dict_pretrained[backbone_nm+k]
    state_dict_loaded.update(state_dict_temp)
    model_.load_state_dict(state_dict_loaded)

    del state_dict_loaded, state_dict_pretrained, state_dict_temp

    return model_