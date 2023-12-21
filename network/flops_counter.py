import os, sys, glob
sys.path.insert(0, os.path.abspath('.'))
import torch
from ptflops import get_model_complexity_info
from network.facexzoo_network import mobilefacenet, efficientnet, resnet_ir_50, swin_transformer
from network.facexzoo_network.efficientnet import efficientnet as efficientnet_b0
from network import gc2sa_net as net

params_dict = {}
macs_dict = {}
embd_dim = 1024

with torch.cuda.device(0):
    # MobileFaceNet
    mobilefacenet = mobilefacenet.MobileFaceNet(embedding_size=embd_dim)
    mobilefacenet_macs, mobilefacenet_params = get_model_complexity_info(mobilefacenet, (3, 112, 112), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', mobilefacenet_macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', mobilefacenet_params))
    params_dict['mobilefacenet'] = mobilefacenet_params
    macs_dict['mobilefacenet'] = mobilefacenet_macs

    # ResNet-50
    resnet = resnet_ir_50.Resnet(num_layers=50, drop_ratio=0)
    resnet_macs, resnet_params = get_model_complexity_info(resnet, (3, 112, 112), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', resnet_macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', resnet_params))
    params_dict['resnet'] = resnet_params
    macs_dict['resnet'] = resnet_macs

    # EfficientNet-b0
    blocks_args, global_params = efficientnet_b0(
                width_coefficient=1.0, depth_coefficient=1.0, 
                dropout_rate=0, image_size=112)
    efficientnet = efficientnet.EfficientNet(7, 7, 512, blocks_args, global_params)
    efficientnet_macs, efficientnet_params = get_model_complexity_info(efficientnet, (3, 112, 112), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', efficientnet_macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', efficientnet_params))
    params_dict['efficientnet'] = efficientnet_params
    macs_dict['efficientnet'] = efficientnet_macs    

    # Swin-T
    depths = [2,2,6,2]
    swint = swin_transformer.SwinTransformer(img_size=224,
                                        patch_size=4,
                                        in_chans=3,
                                        embed_dim=96,
                                        depths=depths,
                                        num_heads=[3,6,12,24],
                                        window_size=7,
                                        mlp_ratio=4.0,
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop_rate=0,
                                        drop_path_rate=0.3,
                                        ape=False,
                                        patch_norm=True,
                                        use_checkpoint=False)
    swint_macs, swint_params = get_model_complexity_info(swint, (3, 224, 224), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', swint_macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', swint_params))
    params_dict['swint'] = swint_params
    macs_dict['swint'] = swint_macs

    # GC2SA-Net
    gc2sa = net.GC2SA_Net(embedding_size=embd_dim)
    gc2sa_macs, gc2sa_params = get_model_complexity_info(gc2sa, (3, 112, 112), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', gc2sa_macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', gc2sa_params))
    params_dict['gc2sa'] = gc2sa_params
    macs_dict['gc2sa'] = gc2sa_macs

    print('Parameters:', params_dict)
    print('MACs', macs_dict)

    torch.save(params_dict, './network/params_dict.pt')
    torch.save(macs_dict, './network/macs_dict.pt')