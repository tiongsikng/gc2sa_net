# *** *** *** ***
# Boiler Codes - Import Dependencies

if __name__ == '__main__': # used for Windows freeze_support() issues
    from torch.optim import lr_scheduler
    import torch.nn as nn
    import torch.optim as optim
    import torch
    import torch.multiprocessing
    import pickle
    import numpy as np
    import random
    import copy
    import os, sys, glob, shutil
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import json
    import argparse
    from torchsummary import summary

    sys.path.insert(0, os.path.abspath('.'))
    from configs.params import *
    from configs import params
    from configs import datasets_config as config
    from data import data_loader as data_loader
    from network.logits import ArcFace, MACL
    import network.gc2sa_net as net
    import train
    from network import load_model
    from eval import verification
    from eval import identification
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Imported.")

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--method', default=params.method, type=str,
                        help='method (backbone)')
    parser.add_argument('--remarks', default=params.remarks, type=str,
                        help='additional remarks')
    parser.add_argument('--write_log', default=params.write_log, type=bool,
                        help='flag to write logs')
    parser.add_argument('--dim', default=params.dim, type=int, metavar='N',
                        help='embedding dimension')
    parser.add_argument('--epochs', default=params.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=params.lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--w_decay', '--w_decay', default=params.w_decay, type=float,
                        metavar='Weight Decay', help='weight decay')
    parser.add_argument('--dropout', '--dropout', default=params.dropout, type=float,
                        metavar='Dropout', help='dropout probability')
    parser.add_argument('--pretrained', default='/home/tiongsik/Python/magnum_opus/models/pretrained/MobileFaceNet_1024.pt', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')

    args = parser.parse_args()

    # Determine if an nvidia GPU is available
    device = params.device

    # For reproducibility, Seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    file_main_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    print('Running on device: {}'.format(device))
    start_ = datetime.now()
    start_string = start_.strftime("%Y%m%d_%H%M%S")

    # *** *** *** ***
    # Load Dataset and Display Other Parameters

    # Face images
    face_train_dir = config.trainingdb['face_train']
    face_loader_train, face_train_set = data_loader.gen_data(face_train_dir, 'train_rand', type='face', aug='True')
    face_loader_train_tl, face_train_tl_set = data_loader.gen_data(face_train_dir, 'train', type='face', aug='True')

    # Periocular Images
    peri_train_dir = config.trainingdb['peri_train']
    peri_loader_train, peri_train_set = data_loader.gen_data(peri_train_dir, 'train_rand', type='periocular', aug='True')
    peri_loader_train_tl, peri_train_tl_set = data_loader.gen_data(peri_train_dir, 'train', type='periocular', aug='True')

    # Validation Periocular (Gallery/Val + Probe/Test)
    peri_val_dir = config.trainingdb['peri_val']
    peri_loader_val, peri_val_set = data_loader.gen_data(peri_val_dir, 'test', type='periocular', aug='False')
    peri_test_dir = config.trainingdb['peri_test']
    peri_loader_test, peri_test_set = data_loader.gen_data(peri_test_dir, 'test', type='periocular', aug='False')

    # Validation Face (Gallery/Val + Probe/Test)
    face_val_dir = config.trainingdb['face_val']
    face_loader_val, face_val_set = data_loader.gen_data(face_val_dir, 'test', type='face', aug='False')
    face_test_dir = config.trainingdb['face_test']
    face_loader_test, face_test_set = data_loader.gen_data(face_test_dir, 'test', type='face', aug='False')

    # Test Periocular (Ethnic)
    ethnic_peri_gallery_dir = config.ethnic['peri_gallery']
    ethnic_peri_probe_dir = config.ethnic['peri_probe']
    ethnic_peri_val_loader, ethnic_peri_val_set = data_loader.gen_data(ethnic_peri_gallery_dir, 'test', type='periocular')
    ethnic_peri_test_loader, ethnic_peri_test_set = data_loader.gen_data(ethnic_peri_probe_dir, 'test', type='periocular')

    # Test Face (Ethnic)
    ethnic_face_gallery_dir = config.ethnic['face_gallery']
    ethnic_face_probe_dir = config.ethnic['face_probe']
    ethnic_face_val_loader, ethnic_face_val_set = data_loader.gen_data(ethnic_face_gallery_dir, 'test', type='face')
    ethnic_face_test_loader, ethnic_face_test_set = data_loader.gen_data(ethnic_face_probe_dir, 'test', type='face')

    # Set and Display all relevant parameters
    print('\n***** Face ( Train ) *****\n')
    face_num_train = len(face_train_set)
    face_num_sub = len(face_train_set.classes)
    print(face_train_set)
    print('Num. of Sub.\t\t:', face_num_sub)
    print('Num. of Train. Imgs (Face) \t:', face_num_train)

    print('\n***** Periocular ( Train ) *****\n')
    peri_num_train = len(peri_train_set)
    peri_num_sub = len(peri_train_set.classes)
    print(peri_train_set)
    print('Num. of Sub.\t\t:', peri_num_sub)
    print('Num. of Train Imgs (Periocular) \t:', peri_num_train)

    print('\n***** Periocular ( Validation (Gallery) ) *****\n')
    peri_num_val = len(peri_val_set)
    print(peri_val_set)
    print('Num. of Sub.\t\t:', len(peri_val_set.classes))
    print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    print('\n***** Periocular ( Validation (Probe) ) *****\n')
    peri_num_test = len(peri_test_set)
    print(peri_test_set)
    print('Num. of Sub.\t\t:', len(peri_test_set.classes))
    print('Num. of Test Imgs (Periocular) \t:', peri_num_test)

    # print('\n***** Face ( Validation (Gallery) ) *****\n')
    # peri_num_val = len(face_val_set)
    # print(face_val_set)
    # print('Num. of Sub.\t\t:', len(face_val_set.classes))
    # print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    # print('\n***** Face ( Test (Probe) ) *****\n')
    # face_num_test = len(face_test_set)
    # print(face_test_set)
    # print('Num. of Sub.\t\t:', len(face_test_set.classes))
    # print('Num. of Test Imgs (Periocular) \t:', face_num_test)

    print('\n***** Other Parameters *****\n')
    print('Start Time \t\t: ', start_string)
    print('Method (Backbone)\t: ', args.method)
    print('Remarks\t\t\t: ', args.remarks)
    print('Net. Descr.\t\t: ', net_descr)
    print('Seed\t\t\t: ', seed)
    print('Batch # Sub.\t\t: ', batch_sub)
    print('Batch # Samp.\t\t: ', batch_samp)
    print('Batch Size\t\t: ', batch_size)
    print('Emb. Dimension\t\t: ', args.dim)
    print('# Epoch\t\t\t: ', epochs)
    print('Learning Rate\t\t: ', args.lr)
    print('LR Scheduler\t\t: ', lr_sch)
    print('Weight Decay\t\t: ', args.w_decay)
    print('Dropout Prob.\t\t: ', args.dropout)
    print('BN Flag\t\t\t: ', bn_flag)
    print('BN Momentum\t\t: ', bn_moment)
    print('Scaling\t\t\t: ', af_s)
    print('Margin\t\t\t: ', af_m)
    print('Save Flag\t\t: ', save)
    print('Log Writing\t\t: ', args.write_log)

    # *** *** *** ***
    # Load Pre-trained Model, Define Loss and Other Hyperparameters for Training

    print('\n***** *****\n')
    print('Loading Pretrained Model' )  
    print()

    train_mode = 'eval'
    model = net.GC2SA_Net(embedding_size = args.dim, do_prob = args.dropout).eval().to(device)

    load_model_path = args.pretrained
    state_dict_loaded = model.state_dict()
    # state_dict_pretrained = torch.load(load_model_path, map_location = device)
    state_dict_pretrained = torch.load(load_model_path, map_location = device)['state_dict']
    state_dict_temp = {}

    for k in state_dict_loaded:
        if 'encoder' not in k:
            # state_dict_temp[k] = state_dict_pretrained[k]
            state_dict_temp[k] = state_dict_pretrained['backbone.'+k]
        else:
            print(k, 'not loaded!')
    state_dict_loaded.update(state_dict_temp)
    model.load_state_dict(state_dict_loaded)
    del state_dict_loaded, state_dict_pretrained, state_dict_temp

    # for multiple GPU usage, set device in params to torch.device('cuda') without specifying GPU ID.
    # model = torch.nn.DataParallel(model).cuda()
    ####

    in_features  = model.linear.in_features
    out_features = args.dim 

    # for MobileFaceNet or GC2SA-Net
    model.linear = nn.Linear(in_features, out_features, bias = True)                      # Deep Embedding Layer
    model.bn = nn.BatchNorm1d(out_features, eps = 1e-5, momentum = 0.1, affine = True) # BatchNorm1d Layer

    #### model summary
    # torch.cuda.empty_cache()
    # import os
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # summary(model.to(device),(3,112,112))

    # *** ***

    # print('\n***** *****\n')
    print('Appending Face-FC to model ( w.r.t. Face ) ... ' )  
    face_fc = ArcFace(in_features = out_features, out_features = face_num_sub, s = af_s, m = af_m, device = device).eval().to(device)

    # *** ****

    # print('\n***** *****\n')
    print('Appending Peri-FC to model ( w.r.t. Periocular ) ... ' )
    peri_fc = copy.deepcopy(face_fc)
    
    # **********

    print('Re-Configuring Model, Face-FC, and Peri-FC ... ' ) 
    print()

    # *** ***
    # model : Determine parameters to be freezed, or unfreezed

    # linear probing in stage 1
    for name, param in model.named_parameters():
        # param.requires_grad = True
        if epochs_pre > 0:
            param.requires_grad = False # Freezing
            if name in ['linear.weight', 'linear.bias', 'bn.weight', 'bn.bias'] or 'encoder' in name:
                param.requires_grad = True # Unfreeze these named layers
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True        

    # model : Display all learnable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('model (requires grad)\t:', name)
            
    # *** 

    # model : Freeze or unfreeze BN parameters
    for name, layer in model.named_modules():
        if isinstance(layer,torch.nn.BatchNorm2d):
            # ***
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True
            # *** 

    if bn_flag == -1:
        print('model\t: EXCLUDE BatchNorm2D Parameters')
    elif bn_flag == 0 or bn_flag == 1:
        print('model\t: INCLUDE BatchNorm2D.weight & bias')

    # *** ***

    # face_fc : Determine parameters to be freezed, or unfreezed
    for param in face_fc.parameters():
        if face_fc_ce_flag is True:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # face_fc : Display all learnable parameters
    print()
    print('face_fc\t:', face_fc)
    for name, param in face_fc.named_parameters():
        if param.requires_grad:   
            print('face_fc\t:', name)

    # *** ***

    # peri_fc : Determine parameters to be freezed, or unfreezed
    for param in peri_fc.parameters():
        if peri_fc_ce_flag is True:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # peri_fc : Display all learnable parameters
    print('peri_fc\t:', peri_fc)
    for name, param in peri_fc.named_parameters():
        if param.requires_grad:
            print('peri_fc\t:', name)

    # ********** 
    # Set an optimizer, scheduler, etc.
    loss_fn = { 'loss_ce' : torch.nn.CrossEntropyLoss(),
                'loss_macl': MACL(temperature=0.07, contrast_mode='all', base_temperature=0.07, device=device, intra_weighting=1.0, inter_weighting=1.25)}
            
    parameters_backbone = [p for p in model.parameters() if p.requires_grad]
    parameters_face_fc = [p for p in face_fc.parameters() if p.requires_grad]
    parameters_peri_fc = [p for p in peri_fc.parameters() if p.requires_grad]

    optimizer = optim.AdamW([   {'params': parameters_backbone},
                                {'params': parameters_face_fc, 'lr': lr*10, 'weight_decay': args.w_decay},
                                {'params': parameters_peri_fc, 'lr': lr*10, 'weight_decay': args.w_decay},
                            ], lr = args.lr, weight_decay = args.w_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_sch, gamma = 0.1)

    metrics = { 'fps': train.BatchTimer(), 'acc': train.accuracy}

    net_params = { 'time' : start_string, 'network' : net_descr, 'method' : args.method, 'remarks' : args.remarks,
                'face_fc_ce_flag' : face_fc_ce_flag, 'peri_fc_ce_flag' : peri_fc_ce_flag, 'face_peri_loss_flag' : face_peri_loss_flag,
                'face_num_sub' : face_num_sub, 'peri_num_sub': peri_num_sub, 'scale' : af_s, 'margin' : af_m,
                'lr' : args.lr, 'lr_sch': lr_sch, 'w_decay' : args.w_decay, 'dropout' : args.dropout,
                'batch_sub' : batch_sub, 'batch_samp' : batch_samp, 'batch_size' : batch_size, 'dims' : args.dim, 'seed' : seed }

    # *** *** *** ***
    #### Model Training

    #### Define Logging
    train_mode = 'train'
    log_folder = "./logs/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks)
    if not os.path.exists(log_folder) and args.write_log is True:
        os.makedirs(log_folder)
    log_nm = log_folder + "/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks) + ".txt"

    # Files Backup
    if args.write_log is True: # only backup if there is log
        # copy main and training files as backup
        for files in glob.glob(os.path.join(file_main_path, '*')):
            if '__' not in files: # ignore __pycache__
                shutil.copy(files, log_folder)
                print(files)
        # networks and logits
        py_extension = '.py'
        desc = file_main_path.split('/')[-1]
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'configs'), 'params' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'gc2sa_net' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'logits' + py_extension), log_folder)

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write(str(net_descr) + "\n")
        file.write('Training started at ' + str(start_) + ".\n\n")
        file.write('Model parameters: \n')
        file.write(json.dumps(net_params) + "\n\n")
        file.close()

    # *** ***
    #### Training

    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0
    peri_best_val_acc = 0

    best_model = copy.deepcopy(model.state_dict())
    best_face_fc = copy.deepcopy(face_fc.state_dict())
    best_peri_fc = copy.deepcopy(peri_fc.state_dict())

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    model.eval().to(device)
    face_fc.eval().to(device)
    peri_fc.eval().to(device)

    #### Test before training

    _, ethnic_peri_test_acc = identification.intramodal_id(model, ethnic_peri_val_loader, ethnic_peri_test_loader, 
                                                                                        device = device, peri_flag = True)
    ethnic_peri_test_acc = np.around(ethnic_peri_test_acc, 4)
    print('Test Rank-1 IR (Ethnic - Intra-Modal (Periocular))\t: ', ethnic_peri_test_acc)   

    _, ethnic_face_test_acc = identification.intramodal_id(model, ethnic_face_val_loader, ethnic_face_test_loader, 
                                                                                    device = device, peri_flag = False)
    ethnic_face_test_acc = np.around(ethnic_face_test_acc, 4)
    print('Test Rank-1 IR (Ethnic - Intra-Modal (Face))\t: ', ethnic_face_test_acc)   

    _, ethnic_cross_peri_acc = identification.intermodal_id(model, ethnic_face_test_loader, ethnic_peri_val_loader,
                                                               device = device, face_model = None, peri_model = None, gallery = 'peri')
    ethnic_cross_peri_acc = np.around(ethnic_cross_peri_acc, 4)
    print('Test Rank-1 IR (Ethnic - Inter-Modal (Periocular))\t: ', ethnic_cross_peri_acc)   

    _, ethnic_cross_face_acc = identification.intermodal_id(model, ethnic_face_val_loader, ethnic_peri_test_loader, 
                                                            device = device, face_model = None, peri_model = None, gallery = 'face')
    ethnic_cross_face_acc = np.around(ethnic_cross_face_acc, 4)
    print('Test Rank-1 IR (Ethnic - Inter-Modal (Face))\t: ', ethnic_cross_face_acc)    
    

    #### Start Training

    for epoch in range(0, 1):    
        print()
        print()        
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train().to(device)
        face_fc.eval().to(device)
        peri_fc.eval().to(device)
        
        if face_fc_ce_flag is True:
            face_fc.train().to(device)    
        if peri_fc_ce_flag is True:
            peri_fc.train().to(device)
        
        # linear probing in stage 2
        if epoch + 1 > epochs_pre:
            for name, param in model.named_parameters():
                param.requires_grad = True
        
        # Use running_stats for training and testing
        if bn_flag != 2:
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):                
                    layer.eval()
                
        train_acc, loss = train.run_train(model, face_fc = face_fc, peri_fc = peri_fc,
                                            face_loader = face_loader_train, peri_loader = peri_loader_train, face_loader_tl = face_loader_train_tl, peri_loader_tl = peri_loader_train_tl,
                                            net_params = net_params, loss_fn = loss_fn, optimizer = optimizer, 
                                            scheduler = scheduler, batch_metrics = metrics, 
                                            show_running = True, device = device, writer = writer)   
        print('Loss : ', loss)

        # *** ***    
        model.eval().to(device)
        face_fc.eval().to(device)
        peri_fc.eval().to(device)
        # *****
        
        # Validation
        # print('Periocular')
        _, peri_val_acc = identification.intramodal_id(model, peri_loader_val, peri_loader_test, device = device, peri_flag = True)
        peri_val_acc = np.around(peri_val_acc, 4)
        print('Validation Rank-1 IR (Periocular)\t: ', peri_val_acc)

        # Testing (Ethnic)
        _, ethnic_peri_test_acc = identification.intramodal_id(model, ethnic_peri_val_loader, ethnic_peri_test_loader, 
                                                                                        device = device, peri_flag = True)
        ethnic_peri_test_acc = np.around(ethnic_peri_test_acc, 4)
        print('Test Rank-1 IR (Ethnic - Periocular)\t: ', ethnic_peri_test_acc)

        _, ethnic_cross_peri_acc = identification.intermodal_id(model, ethnic_face_test_loader, ethnic_peri_val_loader,
                                                            device = device, face_model = None, peri_model = None, gallery = 'peri')
        ethnic_cross_peri_acc = np.around(ethnic_cross_peri_acc, 4)
        print('Test Rank-1 IR (Ethnic Cross - Face)\t: ', ethnic_cross_peri_acc)   
        
        if args.write_log is True:
            file = open(log_nm, 'a+')
            file.write(str('Epoch {}/{}'.format(epoch + 1, epochs)) + "\n")
            file.write('Loss : ' + str(loss) + "\n")
            file.write('Validation Rank-1 IR (Periocular)\t: ' + str(peri_val_acc) + "\n")
            file.write('Test Rank-1 IR (Periocular) \t: ' + str(ethnic_peri_test_acc) + "\n")
            file.write('Test Rank-1 IR (Cross Periocular) \t: ' + str(ethnic_cross_peri_acc) + "\n\n")
            file.close()

        # if save == True:
        if peri_val_acc >= peri_best_val_acc and epoch + 1 >= lr_sch[0] and save == True:            
            best_epoch = epoch + 1
            best_train_acc = train_acc
            peri_best_val_acc = peri_val_acc

            best_model = copy.deepcopy(model.state_dict())
            best_face_fc = copy.deepcopy(face_fc.state_dict())
            best_peri_fc = copy.deepcopy(peri_fc.state_dict())

            print('\n***** *****\n')
            print('Saving Best Model & Rank-1 IR ... ')
            print()
            
            # Set save_best_model_path
            tag = str(args.method) +  '/' + net_tag + '_' + str(batch_sub) + '_' + str(batch_samp) + '/'
            
            save_best_model_dir = './models/best_model/' + tag
            if not os.path.exists(save_best_model_dir):
                os.makedirs(save_best_model_dir)

            save_best_face_fc_dir = './models/best_face_fc/' + tag
            if not os.path.exists(save_best_face_fc_dir):
                os.makedirs(save_best_face_fc_dir)
                
            save_best_peri_fc_dir = './models/best_peri_fc/' + tag
            if not os.path.exists(save_best_peri_fc_dir):
                os.makedirs(save_best_peri_fc_dir)
                
            save_best_acc_dir = './models/best_acc/' + tag
            if not os.path.exists(save_best_acc_dir):
                os.makedirs(save_best_acc_dir)  
                    
            tag = str(args.method) + '_S' + str(af_s) + '_M' + str(af_m) + '_' + str(args.remarks)

            tag = tag + '_BN' + str(bn_flag) + '_M' + str(bn_moment)       

            save_best_model_path = save_best_model_dir + tag + '_' + str(start_string) + '.pth'
            save_best_face_fc_path = save_best_face_fc_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_peri_fc_path = save_best_peri_fc_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_acc_path = save_best_acc_dir + tag + '_' + str(start_string) + '.pkl' 
                    
            print('Best Model Pth\t: ', save_best_model_path)
            print('Best Face-FC Pth\t: ', save_best_face_fc_path)
            print('Best Peri-FC Pth\t: ', save_best_peri_fc_path)
            print('Best Rank-1 IR Pth\t: ', save_best_acc_path)

            # *** ***
            
            torch.save(best_model, save_best_model_path)
            torch.save(best_face_fc, save_best_face_fc_path)
            torch.save(best_peri_fc, save_best_peri_fc_path)

            with open(save_best_acc_path, 'wb') as f:
                pickle.dump([ best_epoch, best_train_acc, peri_best_val_acc ], f)

    if args.write_log is True:
        file = open(log_nm, 'a+')
        end_ = datetime.now()
        end_string = end_.strftime("%Y%m%d_%H%M%S")
        file.write('Training completed at ' + str(end_) + ".\n\n")
        file.write("Model (Path): " + str(save_best_model_path) + "\n\n")    
        file.close()

    # *** *** *** ***
    # Evaluation (Validation)
    print('**** Validation **** \n')
    # Identification - Validation
    print('Periocular - Validation')
    _, peri_val_acc = identification.intramodal_id(model, peri_loader_val, peri_loader_test, device = device, peri_flag = True)
    peri_val_acc = np.around(peri_val_acc, 4)
    print('Validation Rank-1 IR (Periocular)\t: ', peri_val_acc)

    print('Face - Validation')
    _, face_val_acc = identification.intramodal_id(model, face_loader_val, face_loader_test, device = device, peri_flag = False)
    face_val_acc = np.around(face_val_acc, 4)
    print('Validation Rank-1 IR (Face)\t: ', face_val_acc)

    # *** *** *** ***
    #### Identification and Verification for Test Datasets ( Ethnic, Pubfig, FaceScrub, IMDb Wiki, AR)

    print('\n**** Testing Evaluation (All Datasets) **** \n')
    #### Identification (Face and Periocular)
    print("Intra-Modal Rank-1 IR (Periocular) \n")
    peri_id_dict = identification.main_intramodal_id(model, root_pth=config.evaluation['identification'], modal='periocular', peri_flag = True, device = device)
    peri_id_dict = copy.deepcopy(peri_id_dict)
    print(peri_id_dict)

    print("Intra-Modal Rank-1 IR (Face) \n")
    face_id_dict = identification.main_intramodal_id(model, root_pth=config.evaluation['identification'], modal='face', peri_flag = False, device = device)
    face_id_dict = copy.deepcopy(face_id_dict)
    print(face_id_dict)

    #### Verification (Face and Periocular)
    print("Intra-Modal EER (Periocular) \n")
    peri_eer_dict = verification.intramodal_verify(model, out_features, root_drt = config.evaluation['verification'], peri_flag = True, device = device)
    peri_eer_dict = copy.deepcopy(peri_eer_dict)
    print(peri_eer_dict)

    print("Intra-Modal EER (Face) \n")
    face_eer_dict = verification.intramodal_verify(model, out_features, root_drt = config.evaluation['verification'], peri_flag = False, device = device)
    face_eer_dict = copy.deepcopy(face_eer_dict)
    print(face_eer_dict)

    #### Inter-Modal Identification (Face and Periocular)
    print("Inter-Modal Rank-1 IR (Periocular-Face) \n")
    inter_id_dict_f, inter_id_dict_p= identification.main_intermodal_id(model, root_pth = config.evaluation['identification'], device = device)
    inter_id_dict_f, inter_id_dict_p = copy.deepcopy(inter_id_dict_f), copy.deepcopy(inter_id_dict_p)
    print(inter_id_dict_p, inter_id_dict_f)

    #### Inter-Modal Verification (Face and Periocular)
    print("Inter-Modal EER (Periocular-Face) \n")
    inter_eer_dict = verification.intermodal_verify(model, face_model = None, peri_model = None, emb_size = out_features, root_drt = config.evaluation['verification'], device = device)    
    inter_eer_dict = copy.deepcopy(inter_eer_dict)
    print(inter_eer_dict)

    # *** *** *** ***
    # Dataset Performance Summary
    print('**** Testing Summary Results (All Datasets) **** \n')

    # *** ***
    print('\n\n Ethnic \n')

    ethnic_acc_peri = peri_id_dict['ethnic']
    ethnic_eer_peri = peri_eer_dict['ethnic']
    ethnic_acc_face = face_id_dict['ethnic']
    ethnic_eer_face = face_eer_dict['ethnic']
    ethnic_inter_acc_p = inter_id_dict_p['ethnic']
    ethnic_inter_acc_f = inter_id_dict_f['ethnic']
    ethnic_inter_eer = inter_eer_dict['ethnic']

    print('Rank-1 IR (Periocular)\t: ', ethnic_acc_peri)
    print("EER (Periocular)\t: ", ethnic_eer_peri)
    print('Rank-1 IR (Face)\t: ', ethnic_acc_face)
    print("EER (Face)\t: ", ethnic_eer_face)
    print('Inter-Modal Rank-1 IR - Periocular Gallery\t: ', ethnic_inter_acc_p)
    print('Inter-Modal Rank-1 IR - Face Gallery\t: ', ethnic_inter_acc_f)
    print('Inter-Modal EER \t: ', ethnic_inter_eer)


    # *** ***
    print('\n\n Pubfig \n')

    pubfig_acc_peri = peri_id_dict['pubfig']
    pubfig_eer_peri = peri_eer_dict['pubfig']
    pubfig_acc_face = face_id_dict['pubfig']
    pubfig_eer_face = face_eer_dict['pubfig']
    pubfig_inter_acc_p = inter_id_dict_p['pubfig']
    pubfig_inter_acc_f = inter_id_dict_f['pubfig']
    pubfig_inter_eer = inter_eer_dict['pubfig']

    print('Rank-1 IR (Periocular)\t: ', pubfig_acc_peri)
    print("EER (Periocular)\t: ", pubfig_eer_peri)
    print('Rank-1 IR (Face)\t: ', pubfig_acc_face)
    print("EER (Face)\t: ", pubfig_eer_face)
    print('Inter-Modal Rank-1 IR - Periocular Gallery\t: ', pubfig_inter_acc_p)
    print('Inter-Modal Rank-1 IR - Face Gallery\t: ', pubfig_inter_acc_f)
    print('Inter-Modal EER \t: ', pubfig_inter_eer)


    # *** ***
    print('\n\n FaceScrub\n')

    facescrub_acc_peri = peri_id_dict['facescrub']
    facescrub_eer_peri = peri_eer_dict['facescrub']
    facescrub_acc_face = face_id_dict['facescrub']
    facescrub_eer_face = face_eer_dict['facescrub']
    facescrub_inter_acc_p = inter_id_dict_p['facescrub']
    facescrub_inter_acc_f = inter_id_dict_f['facescrub']
    facescrub_inter_eer = inter_eer_dict['facescrub']

    print('Rank-1 IR (Periocular)\t: ', facescrub_acc_peri)
    print("EER (Periocular)\t: ", facescrub_eer_peri)
    print('Rank-1 IR (Face)\t: ', facescrub_acc_face)
    print("EER (Face)\t: ", facescrub_eer_face)
    print('Inter-Modal Rank-1 IR - Periocular Gallery\t: ', facescrub_inter_acc_p)
    print('Inter-Modal Rank-1 IR - Face Gallery\t: ', facescrub_inter_acc_f)
    print('Inter-Modal EER \t: ', facescrub_inter_eer)


    # *** *** *** ***
    print('\n\n IMDB Wiki \n')

    imdb_wiki_acc_peri = peri_id_dict['imdb_wiki']
    imdb_wiki_eer_peri = peri_eer_dict['imdb_wiki']
    imdb_wiki_acc_face = face_id_dict['imdb_wiki']
    imdb_wiki_eer_face = face_eer_dict['imdb_wiki']
    imdb_wiki_inter_acc_p = inter_id_dict_p['imdb_wiki']
    imdb_wiki_inter_acc_f = inter_id_dict_f['imdb_wiki']
    imdb_wiki_inter_eer = inter_eer_dict['imdb_wiki']

    print('Rank-1 IR (Periocular)\t: ', imdb_wiki_acc_peri)
    print("EER (Periocular)\t: ", imdb_wiki_eer_peri)
    print('Rank-1 IR (Face)\t: ', imdb_wiki_acc_face)
    print("EER (Face)\t: ", imdb_wiki_eer_face)
    print('Inter-Modal Rank-1 IR - Periocular Gallery\t: ', imdb_wiki_inter_acc_p)
    print('Inter-Modal Rank-1 IR - Face Gallery\t: ', imdb_wiki_inter_acc_f)
    print('Inter-Modal EER \t: ', imdb_wiki_inter_eer)


    # *** *** *** ***
    print('\n\n AR \n')

    ar_acc_peri = peri_id_dict['ar']
    ar_eer_peri = peri_eer_dict['ar']
    ar_acc_face = face_id_dict['ar']
    ar_eer_face = face_eer_dict['ar']
    ar_inter_acc_p = inter_id_dict_p['ar']
    ar_inter_acc_f = inter_id_dict_f['ar']
    ar_inter_eer = inter_eer_dict['ar']

    print('Rank-1 IR (Periocular)\t: ', ar_acc_peri)
    print("EER (Periocular)\t: ", ar_eer_peri)
    print('Rank-1 IR (Face)\t: ', ar_acc_face)
    print("EER (Face)\t: ", ar_eer_face)
    print('Inter-Modal Rank-1 IR - Periocular Gallery\t: ', ar_inter_acc_p)
    print('Inter-Modal Rank-1 IR - Face Gallery\t: ', ar_inter_acc_f)
    print('Inter-Modal EER \t: ', ar_inter_eer)

    # *** *** *** ***
    #### Average of all Datasets
    print('\n\n\n Calculating Average \n')

    avg_peri_ir = identification.get_avg(peri_id_dict)
    avg_face_ir = identification.get_avg(face_id_dict)
    avg_inter_p_ir = identification.get_avg(inter_id_dict_p)
    avg_inter_f_ir = identification.get_avg(inter_id_dict_f)
    avg_peri_eer = verification.get_avg(peri_eer_dict)
    avg_face_eer = verification.get_avg(face_eer_dict)
    avg_inter_eer = verification.get_avg(inter_eer_dict)

    print('Rank-1 IR (Periocular)\t: ', avg_peri_ir['avg'])
    print('Rank-1 IR (Face)\t: ', avg_face_ir['avg'])
    print('Inter-Modal Rank-1 IR - Periocular Gallery\t: ', avg_inter_p_ir['avg'])
    print('Inter-Modal Rank-1 IR - Face Gallery\t: ', avg_inter_f_ir['avg'])
    print("EER (Periocular)\t: ", avg_peri_eer['avg'])
    print("EER (Face)\t: ", avg_face_eer['avg'])
    print('Inter-Modal EER \t: ', avg_inter_eer['avg'])


    # *** *** *** ***
    # Write Final Performance Summaries to Log 

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write('****Ethnic:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['ethnic']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['ethnic']))
        file.write('\n\nInter-Modal (ID): \n Periocular Gallery - ' + str(inter_id_dict_p['ethnic']) + ', \n Face Gallery - ' + str(inter_id_dict_f['ethnic']) + '\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['ethnic']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ethnic']))        
        file.write('\n\nInter-Modal (Ver): ' + str(inter_eer_dict['ethnic']) + '\n\n\n')

        file.write('****Pubfig:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['pubfig']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['pubfig']))
        file.write('\n\nInter-Modal (ID): \n Periocular Gallery - ' + str(inter_id_dict_p['pubfig']) + ', \n Face Gallery - ' + str(inter_id_dict_f['pubfig']) + '\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['pubfig']) + '\nFinal EER. (Face): ' + str(face_eer_dict['pubfig']))    
        file.write('\n\nInter-Modal (Ver): ' + str(inter_eer_dict['pubfig']) + '\n\n\n')

        file.write('****FaceScrub:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['facescrub']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['facescrub']))
        file.write('\n\nInter-Modal (ID): \n Periocular Gallery - ' + str(inter_id_dict_p['facescrub']) + ', \n Face Gallery - ' + str(inter_id_dict_f['facescrub']) + '\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['facescrub']) + '\nFinal EER. (Face): ' + str(face_eer_dict['facescrub']))        
        file.write('\n\nInter-Modal (Ver): ' + str(inter_eer_dict['facescrub']) + '\n\n\n')

        file.write('****IMDB Wiki:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['imdb_wiki']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['imdb_wiki']))
        file.write('\n\nInter-Modal (ID): \n Periocular Gallery - ' + str(inter_id_dict_p['imdb_wiki']) + ', \n Face Gallery - ' + str(inter_id_dict_f['imdb_wiki']) + '\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['imdb_wiki']) + '\nFinal EER. (Face): ' + str(face_eer_dict['imdb_wiki']))        
        file.write('\n\nInter-Modal (Ver): ' + str(inter_eer_dict['imdb_wiki']) + '\n\n\n')

        file.write('****AR:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['ar']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['ar']))
        file.write('\n\nInter-Modal (ID): \n Periocular Gallery - ' + str(inter_id_dict_p['ar']) + ', \n Face Gallery - ' + str(inter_id_dict_f['ar']) + '\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['ar']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ar']))        
        file.write('\n\nInter-Modal (Ver): ' + str(inter_eer_dict['ar']) + '\n\n\n')

        file.write('\n\n **** Average **** \n\n')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(avg_peri_ir['avg']) + '\nFinal Test Rank-1 IR (Face): ' + str(avg_face_ir['avg']))
        file.write('\n\nInter-Modal (ID): \n Periocular Gallery - ' + str(avg_inter_p_ir['avg']) + ', \n Face Gallery - ' + str(avg_inter_f_ir['avg']) + '\n')
        file.write('\nFinal EER. (Periocular): ' + str(avg_peri_eer['avg']) + '\nFinal EER. (Face): ' + str(avg_face_eer['avg']))        
        file.write('\n\nInter-Modal (Ver): ' + str(avg_inter_eer['avg']) + '\n\n\n')
        file.close()
    # *** *** *** ***                 