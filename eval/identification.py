import os, sys, glob, copy
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import time
import torch
import torch.utils.data
from torch.nn import functional as F
from data import data_loader
from sklearn.metrics import pairwise
from sklearn.model_selection import KFold
import network.gc2sa_net as net
from network import load_model
from configs import datasets_config as config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

id_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
inter_id_dict_f = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
inter_id_dict_p = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar'] 


def get_avg(dict_list):
    total_ir = 0
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    for items in dict_list:
        total_ir += dict_list[items]
    dict_list['avg'] = total_ir/len(dict_list)

    return dict_list

# Intra-Modal Identification (Main)
def main_intramodal_id(model, root_pth=config.evaluation['identification'], modal='periocular', peri_flag = True, device = 'cuda:0'):
    print('Modal:', modal[:4])

    for datasets in dset_list:
        root_drt = root_pth + datasets + '/**'        
        modal_root = '/' + modal[:4] + '/'
        probe_data_loaders = []
        probe_data_sets = []
        acc = []        

        # data loader and datasets
        for directs in glob.glob(root_drt):
            base_nm = directs.split('\\')[-1]
            modal_base = directs + modal_root
            if not datasets in ['ethnic']:
                if modal_base.split('/')[-3] != 'gallery':
                    data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                    probe_data_loaders.append(data_load)
                    probe_data_sets.append(data_set)
                else: 
                    data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                    gallery_data_loaders = data_load
                    gallery_data_sets = data_set
        # *** ***

        if datasets == 'ethnic':
            ethnic_gal_data_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            ethnic_pr_data_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            _, acc = intramodal_id(model, ethnic_gal_data_load, ethnic_pr_data_load, device = device, peri_flag = peri_flag)

        else:
                for i in range(len(probe_data_loaders)):
                    _, test_acc = intramodal_id(model, gallery_data_loaders, probe_data_loaders[i], 
                                                                                                device = device, peri_flag = peri_flag)
                    test_acc = np.around(test_acc, 4)
                    acc.append(test_acc)

        # *** ***

        acc = np.around(np.mean(acc), 4)
        print(datasets, acc)
        id_dict[datasets] = acc

    return id_dict


# Inter-Modal Identification (Main)
def main_intermodal_id(model, root_pth=config.evaluation['identification'], face_model = None, peri_model = None, device = 'cuda:0'):
    for datasets in dset_list:

        root_drt = root_pth + datasets + '/**'
        modal_root = ['/peri/', '/face/']
        path_lst = []
        acc_face_gal = []
        acc_peri_gal = []    

        # *** ***

        if datasets == 'ethnic':
            ethnic_face_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', 'face', aug='False')
            ethnic_peri_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', 'periocular', aug='False')
            _, inter_face_gal_acc_ethnic = intermodal_id(model, ethnic_face_gal_load, ethnic_peri_pr_load, device = device, face_model = face_model, peri_model = peri_model, gallery = 'face')
            inter_face_gal_acc_ethnic = np.around(inter_face_gal_acc_ethnic, 4)
            acc_face_gal.append(inter_face_gal_acc_ethnic)

            ethnic_peri_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', 'periocular', aug='False')
            ethnic_face_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', 'face', aug='False')
            _, inter_peri_gal_acc_ethnic = intermodal_id(model, ethnic_face_pr_load, ethnic_peri_gal_load, device = device, face_model = face_model, peri_model = peri_model, gallery = 'peri')
            inter_peri_gal_acc_ethnic = np.around(inter_peri_gal_acc_ethnic, 4)
            acc_peri_gal.append(inter_peri_gal_acc_ethnic)

        else:
            # data loader and datasets
            for directs in glob.glob(root_drt):
                if not directs.split('/')[-1] == 'gallery':
                    path_lst.append(directs)
                else:
                    gallery_path = directs      

            for probes in path_lst:
                peri_probe_load, peri_dataset = data_loader.gen_data((probes + modal_root[0]), 'test', 'periocular', aug='False')
                face_gal_load, face_dataset = data_loader.gen_data((gallery_path + modal_root[1]), 'test', 'face', aug='False')
                _, inter_face_gal_acc = intermodal_id(model, face_gal_load, peri_probe_load, device = device, face_model = face_model, peri_model = peri_model, gallery = 'face')
                inter_face_gal_acc = np.around(inter_face_gal_acc, 4)
                acc_face_gal.append(inter_face_gal_acc)

                peri_gal_load, peri_dataset = data_loader.gen_data((gallery_path + modal_root[0]), 'test', 'periocular', aug='False')
                face_probe_load, face_dataset = data_loader.gen_data((probes + modal_root[1]), 'test', 'face', aug='False')
                _, inter_peri_gal_acc = intermodal_id(model, face_probe_load, peri_gal_load, device = device, face_model = face_model, peri_model = peri_model, gallery = 'peri')
                inter_peri_gal_acc = np.around(inter_peri_gal_acc, 4)
                acc_peri_gal.append(inter_peri_gal_acc)

        # *** ***

        acc_peri_gal = np.around(np.mean(acc_peri_gal), 4)
        acc_face_gal = np.around(np.mean(acc_face_gal), 4)        
        print('Peri Gallery:', datasets, acc_peri_gal)
        print('Face Gallery:', datasets, acc_face_gal) 
        inter_id_dict_p[datasets] = acc_peri_gal       
        inter_id_dict_f[datasets] = acc_face_gal        

    return inter_id_dict_f, inter_id_dict_p


# Intra-Modal Identification Function
def intramodal_id(model, loader_gallery, loader_test, device = 'cuda:0', peri_flag = False):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False
        
    # ***** *****
    
    # Extract gallery features w.r.t. pre-learned model
    gallery_fea = torch.tensor([])
    gallery_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_gallery):

            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

            gallery_fea = torch.cat((gallery_fea, x.detach().cpu()), 0)
            gallery_label = torch.cat((gallery_label, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(gallery_fea.size()[0] == gallery_label.size()[0])
    
    del loader_gallery
    time.sleep(0.0001)
    
    # ***** *****
    
    # Extract test features w.r.t. pre-learned model
    test_fea = torch.tensor([])
    test_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_test):

            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

            test_fea = torch.cat((test_fea, x.detach().cpu()), 0)
            test_label = torch.cat((test_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    # print('Test Set Capacity\t: ', test_fea.size())
    assert(test_fea.size()[0] == test_label.size()[0])
    
    del loader_test
    time.sleep(0.0001)

    # ***** *****
        
    # Calculate gallery_acc and test_acc
    gallery_label = np.reshape(np.array(gallery_label), -1)
    test_label = np.reshape(np.array(test_label), -1)
    
    gallery_dist = pairwise.cosine_similarity(gallery_fea)
    gallery_pred = np.argmax(gallery_dist, 0)
    gallery_pred = gallery_label[gallery_pred] 
    gallery_acc = sum(gallery_label == gallery_pred) / gallery_label.shape[0]
    
    test_dist = pairwise.cosine_similarity(gallery_fea, test_fea)
    test_pred = np.argmax(test_dist, 0)
    test_pred = gallery_label[test_pred]
    test_acc = sum(test_label == test_pred) / test_label.shape[0]

    # torch.cuda.empty_cache()
    # time.sleep(0.0001)
    
    del model
    time.sleep(0.0001)
    
    return gallery_acc, test_acc


# Inter-Modal Identification Function
def intermodal_id(model, face_loader, peri_loader, device = 'cuda:0', face_model = None, peri_model = None, gallery = 'face'):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False

    # ***** *****

    # Extract face features w.r.t. pre-learned model
    face_fea = torch.tensor([])
    face_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag = False)
            else:
                x = model(x, peri_flag=False)

            face_fea = torch.cat((face_fea, x.detach().cpu()), 0)
            face_label = torch.cat((face_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    # print('Test Set Capacity\t: ', test_fea.size())
    assert(face_fea.size()[0] == face_label.size()[0])
    
    del face_loader
    time.sleep(0.0001)

    # *****    
    
    # Extract periocular features w.r.t. pre-learned model
    peri_fea = torch.tensor([])
    peri_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(peri_loader):

            x = x.to(device)
            if not peri_model is None:
                peri_model = peri_model.eval().to(device)
                x = peri_model(x, peri_flag=True)
            else:
                x = model(x, peri_flag=True)

            peri_fea = torch.cat((peri_fea, x.detach().cpu()), 0)
            peri_label = torch.cat((peri_label, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(peri_fea.size()[0] == peri_label.size()[0])
    
    del peri_loader
    time.sleep(0.0001)
    
    # ***** *****

    # perform checking
    if gallery == 'face':
        gal_fea, gal_label = face_fea, face_label
        probe_fea, probe_label = peri_fea, peri_label
    elif gallery == 'peri':
        gal_fea, gal_label = peri_fea, peri_label
        probe_fea, probe_label = face_fea, face_label

    # normalize features
    gal_fea = F.normalize(gal_fea, p=2, dim=1)
    probe_fea = F.normalize(probe_fea, p=2, dim=1)

    # Calculate gallery_acc and test_acc
    gal_label = np.reshape(np.array(gal_label), -1)
    probe_label = np.reshape(np.array(probe_label), -1)    
    
    gal_dist = pairwise.cosine_similarity(gal_fea)
    gal_pred = np.argmax(gal_dist, 0)
    gal_pred = gal_label[gal_pred] 
    gal_acc = sum(gal_label == gal_pred) / gal_label.shape[0]
    
    probe_dist = pairwise.cosine_similarity(gal_fea, probe_fea)
    probe_pred = np.argmax(probe_dist, 0)
    probe_pred = gal_label[probe_pred]
    probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]
    
    del model
    time.sleep(0.0001)
    
    return gal_acc, probe_acc


if __name__ == '__main__':
    embd_dim = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    load_model_path = './models/best_model/GC2SA-Net.pth'
    model = net.GCCSA_Net(embedding_size = embd_dim).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)

    peri_id_dict = main_intramodal_id(model, root_pth=config.evaluation['identification'], modal = 'periocular', peri_flag = True, device = device)
    peri_id_dict = get_avg(peri_id_dict)
    peri_id_dict = copy.deepcopy(peri_id_dict)
    print('Average (Periocular):', peri_id_dict['avg'])
    print('Intra-Modal (Periocular):', peri_id_dict)    

    face_id_dict = main_intramodal_id(model, root_pth = config.evaluation['identification'], modal = 'face', peri_flag = False, device = device)
    face_id_dict = get_avg(face_id_dict)
    face_id_dict = copy.deepcopy(face_id_dict)
    print('Average (Face):', face_id_dict['avg'])
    print('Intra-Modal (Face):', face_id_dict)    

    inter_id_dict_f, inter_id_dict_p = main_intermodal_id(model, root_pth = config.evaluation['identification'], face_model = None, peri_model = None, device = device)
    inter_id_dict_p, inter_id_dict_f = get_avg(inter_id_dict_p), get_avg(inter_id_dict_f)
    inter_id_dict_p = copy.deepcopy(inter_id_dict_p)
    inter_id_dict_f = copy.deepcopy(inter_id_dict_f)
    print('Average (Periocular-Face):', inter_id_dict_p['avg'], inter_id_dict_f['avg'])
    print('Inter-Modal (Periocular Gallery, Face Gallery):', inter_id_dict_p, inter_id_dict_f)    