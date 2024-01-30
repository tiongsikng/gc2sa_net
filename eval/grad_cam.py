import os, sys, glob
import torch
from tqdm import tqdm
from skimage.io import imread
from torch.nn import Module
import numpy as np
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, to_pil_image, to_tensor#, resize
from PIL import Image
from torchvision import datasets, transforms
from torchcam.utils import overlay_mask
sys.path.insert(0, os.path.abspath('.'))
import network.gc2sa_net as net
from data import data_loader
from network import load_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #

class GradCamModel(Module):
    def __init__(self, model, eval_layer):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.model = model

        self.layerhook.append(eval_layer.register_forward_hook(self.forward_hook()))
        
        for p in self.model.parameters():
            p.requires_grad = True
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out
    

def plot_gradcam(gradcam_model, eval_layer, image_path, root_dir='./graphs/gradcam', method='', modal='peri', base_img_pixels=(112, 112), aleph=0.9):
    from torchvision.transforms.functional import resize
    gradcam_model = SmoothGradCAMpp(model, target_layer = eval_layer, input_shape=(3, 112, 112))
    
    new_img_name = os.path.join(root_dir, method, modal)
    transform = transforms.Compose([    transforms.Normalize(
                                    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
                                    std=[1/0.5, 1/0.5, 1/0.5]),
                                    transforms.ToPILImage()
                               ])
    
    data_load, data_set = data_loader.gen_data(image_path, 'test', type=modal, aug='False')

    for i, (img_in) in tqdm(enumerate(data_set)):
        img_path = data_set.imgs[i][0]
        img_name = img_path.split('/')[-1].split('.')[0]
        pil_img = resize(Image.open(img_path), (112, 112))
        img_tensor = normalize(to_tensor(pil_img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        scores = model(img_tensor.unsqueeze(0).to(device))
        scores = scores[0]
        class_idx = scores.argmax().item()
        activation_map = gradcam_model(img_in[1], scores)
        activation_map = [activation_map[0].detach().cpu()]
        if modal == 'peri':
            base_img_pixels = (37, 112)
        result = overlay_mask(resize(pil_img, base_img_pixels), resize(to_pil_image(activation_map[0].squeeze(0), mode='F'), base_img_pixels), alpha=aleph)

        path = os.path.join(new_img_name, (img_path.split('/')[-2] + '_' + img_name))
        resize(pil_img, base_img_pixels).save(path + '_ori.png', 'PNG')
        result.save(path + '_c.png', 'PNG')


if __name__ == '__main__':
    '''
        root dir: main directory to store all images. Store them in a single folder, e.g., '/home/tiongsik/Python/gc2sa_net/data/gradcam_imgs/[face/peri]/1/'
        method: subfolder for method
        modal: periocular or face
        pixel_size: size of image in pixels
        aleph: opacity of the heatmap shown, 1 = least opaque, 0 = most opaque
        image_path: change image path, create a loop if to be used with multiple images
    '''

    root_dir = './graphs/gradcam'
    method = 'gc2sa_net'
    modal = ['face','peri']
    pixel_size = (112, 112)
    aleph = 0.5

    for modality in modal:
        modality = modality[:4]
        if not os.path.exists(os.path.join(root_dir, str(method), modality)):
            os.makedirs(os.path.join(root_dir, str(method), modality))
    
    # load model and set evaluation layer for GradCAM
    embd_dim = 1024
    load_model_path = './models/best_model/GC2SA-Net.pth'
    model = net.GC2SA_Net(embedding_size = embd_dim).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)
    eval_layer = model.conv_6_sep
            
    for modality in modal:
        image_path = '/home/tiongsik/Python/conditional_biometrics/data/gradcam_imgs/' + str(modality) + '/1/'
        plot_gradcam(model, eval_layer, image_path, root_dir=root_dir,
                method=str(method), modal=modality, base_img_pixels=pixel_size, 
                aleph=(1-aleph))
            