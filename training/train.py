import numpy as np
import time
import sys
import itertools 
import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F

from sklearn.metrics import pairwise

# **********    

class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x
        self.fn_no_mean = lambda x, i: x

    def __call__(self, loss, peri_loss_ce, face_loss_ce, macl_loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:5.4f} | '.format(self.fn(loss, i))
        peri_loss_ce = 'peri_loss: {:5.4f} | '.format(self.fn_no_mean(peri_loss_ce, i))
        face_loss_ce = 'face_loss: {:5.4f} | '.format(self.fn_no_mean(face_loss_ce, i))
        macl_loss = 'macl_loss: {:5.4f} | '.format(self.fn_no_mean(macl_loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + peri_loss_ce + face_loss_ce + macl_loss + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')

# **********

class BatchTimer(object):
    
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    
    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)

# **********

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()
    
# **********

def plot_graph(mixed_ST, nrow=8):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import cv2
    def convert_imgs(imgs):
        for i in range(imgs.shape[0]):
            imgs[i] = imgs[i] #torch.from_numpy(cv2.cvtColor(imgs[i].cpu().detach().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB).transpose(2,0,1))
        return imgs

    # show_batch(mixed_query, 32)
    imgs = convert_imgs(mixed_ST)
    grid_img = make_grid(imgs, nrow=nrow).cpu().detach()
    fig=plt.figure()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

# **********

def run_train(model, face_fc, peri_fc, 
                face_loader, peri_loader, face_loader_tl, peri_loader_tl, 
                epoch = 1, net_params = None, loss_fn = None, optimizer = None, 
                scheduler = None, batch_metrics = {'time': BatchTimer()}, 
                show_running = True, device = 'cuda:0', writer = None):
    
    mode = 'Train'
    iter_max = len(face_loader)
    logger = Logger(mode, length = iter_max, calculate_mean = show_running)
    
    loss = 0
    metrics = {}
    
    # **********
    
    face_iterator_tl = iter(face_loader_tl)
    peri_iterator_tl = iter(peri_loader_tl)

    for batch_idx, ( face_in, peri_in ) in enumerate( zip( face_loader, peri_loader ) ):
        #### *** non-target : face ***
        #### random sampling
        face_in, face_aug_in = face_in

        face_x, face_y = face_in
        face_x_aug, face_y_aug = face_aug_in

        face_x = face_x.to(device)
        face_y = face_y.to(device)
        face_x_aug = face_x_aug.to(device)
        face_y_aug = face_y_aug.to(device)

        del face_in, face_aug_in

        #### balanced sampling
        try:
            face_in_tl, face_aug_in_tl = next(face_iterator_tl)
        except StopIteration:
            face_iterator_tl = iter(face_loader_tl)
            face_in_tl, face_aug_in_tl = next(face_iterator_tl)
        
        face_x_tl, face_y_tl = face_in_tl
        face_x_aug_tl, face_y_aug_tl = face_aug_in_tl

        face_x_tl = face_x_tl.to(device)
        face_y_tl = face_y_tl.to(device)
        face_x_aug_tl = face_x_aug_tl.to(device)
        face_y_aug_tl = face_y_aug_tl.to(device)

        del face_in_tl, face_aug_in_tl
        
        # *** ***
        
        face_emb_r = model(torch.cat((face_x, face_x_aug), dim=0), peri_flag=False)
        face_lbl_r = torch.cat((face_y, face_y_aug))

        del face_x, face_x_aug

        face_emb_tl = model(torch.cat((face_x_tl, face_x_aug_tl), dim=0), peri_flag=False)
        face_lbl_tl = torch.cat((face_y_tl, face_y_aug_tl))    

        del face_x_tl, face_x_aug_tl

        # determine random or balanced sampling for embeddings to be used for CE loss
        face_emb = face_emb_r
        face_lbl = face_lbl_r
        # face_emb = face_emb_tl
        # face_lbl = face_lbl_tl
        # face_emb = torch.cat((face_emb_r, face_emb_tl), dim=0)
        # face_lbl = torch.cat((face_lbl_r, face_lbl_tl))

        # ***

        face_loss_ce = 0
        if net_params['face_fc_ce_flag'] is True:
            face_pred = face_fc(face_emb, face_lbl)
            face_loss_ce = loss_fn['loss_ce'](face_pred, face_lbl)  
        else:
            face_pred = None
        
        #
        # *** ***
        #

        #### *** target : periocular ***
        #### random sampling
        peri_in, peri_aug_in = peri_in

        peri_x, peri_y = peri_in
        peri_x_aug, peri_y_aug = peri_aug_in

        peri_x = peri_x.to(device)
        peri_y = peri_y.to(device)
        peri_x_aug = peri_x_aug.to(device)
        peri_y_aug = peri_y_aug.to(device)

        del peri_in, peri_aug_in

        #### balanced sampling
        try:
            peri_in_tl, peri_aug_in_tl = next(peri_iterator_tl)
        except StopIteration:
            peri_iterator_tl = iter(peri_loader_tl)
            peri_in_tl, peri_aug_in_tl = next(peri_iterator_tl)
        
        peri_x_tl, peri_y_tl = peri_in_tl
        peri_x_aug_tl, peri_y_aug_tl = peri_aug_in_tl

        peri_x_tl = peri_x_tl.to(device)
        peri_y_tl = peri_y_tl.to(device)
        peri_x_aug_tl = peri_x_aug_tl.to(device)
        peri_y_aug_tl = peri_y_aug_tl.to(device)

        del peri_in_tl, peri_aug_in_tl

        # *** ***

        peri_emb_r  = model(torch.cat((peri_x, peri_x_aug), dim=0), peri_flag=True)
        peri_lbl_r = torch.cat((peri_y, peri_y_aug))

        del peri_x, peri_x_aug

        peri_emb_tl = model(torch.cat((peri_x_tl, peri_x_aug_tl), dim=0), peri_flag=True)
        peri_lbl_tl = torch.cat((peri_y_tl, peri_y_aug_tl))

        del peri_x_tl, peri_x_aug_tl
        
        # determine random or balanced sampling for embeddings to be used for CE loss
        peri_emb = peri_emb_r
        peri_lbl = peri_lbl_r
        # peri_emb = peri_emb_tl
        # peri_lbl = peri_lbl_tl
        # peri_emb = torch.cat((peri_emb_r, peri_emb_tl), dim=0)
        # peri_lbl = torch.cat((peri_lbl_r, peri_lbl_tl))

        # ***

        peri_loss_ce = 0
        if net_params['peri_fc_ce_flag'] is True:
            peri_pred = peri_fc(peri_emb, peri_lbl)
            peri_loss_ce = loss_fn['loss_ce'](peri_pred, peri_lbl)
        else:
            peri_pred = None

        # *** *** 
        
        #### MACL (MUST use Balanced Sampling)
        features = torch.cat([face_emb_tl[:int(face_emb_tl.shape[0]/2)].unsqueeze(1), face_emb_tl[int(face_emb_tl.shape[0]/2):].unsqueeze(1), \
                    peri_emb_tl[:int(peri_emb_tl.shape[0]/2)].unsqueeze(1), peri_emb_tl[int(peri_emb_tl.shape[0]/2):].unsqueeze(1)], dim=1)
        labels = face_lbl_tl[:int(face_lbl_tl.shape[0]/2)]

        assert(torch.all(face_lbl_tl[:int(face_lbl_tl.shape[0]/2)] == face_lbl_tl[int(face_lbl_tl.shape[0]/2):]) and \
                torch.all(peri_lbl_tl[:int(peri_lbl_tl.shape[0]/2)] == peri_lbl_tl[int(peri_lbl_tl.shape[0]/2):]) and \
                    torch.all(face_lbl_tl[:int(face_lbl_tl.shape[0]/2)] == peri_lbl_tl[int(peri_lbl_tl.shape[0]/2):]))

        if net_params['face_fc_ce_flag'] is True and net_params['peri_fc_ce_flag'] is True and net_params['face_peri_loss_flag'] is True:
            macl_loss = loss_fn['loss_macl'](features, labels)      

                    
        del face_emb, peri_emb, face_emb_tl, peri_emb_tl, face_emb_r, peri_emb_r 
        
        # *** ***
                        
        # Define loss_batch
        loss_batch = peri_loss_ce + face_loss_ce + macl_loss        

        # *** ***

        # if model.training:
        optimizer.zero_grad()
        loss_batch.backward() 
        optimizer.step()
        
        time.sleep(0.0001)
        
        # *** ***
        
        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            if not peri_pred is None:
                metrics_batch[metric_name] = metric_fn(peri_pred, peri_lbl).detach().cpu()
            elif not face_pred is None:
                metrics_batch[metric_name] = metric_fn(face_pred, face_lbl).detach().cpu()
            else:
                raise ValueError('Neither face nor periocular initialized.')
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
            
        if writer is not None: # and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, peri_loss_ce, face_loss_ce, macl_loss, metrics, batch_idx)
        else:
            logger(loss_batch, metrics_batch, batch_idx)
            
    # END FOR
    # Completed processing for all batches (training and testing), i.e., an epoch 
    
    # *** ***
        
    # if model.training and scheduler is not None:
    if scheduler is not None:
        scheduler.step()

    loss = loss / (batch_idx + 1)
    metrics = {k: v / (batch_idx + 1) for k, v in metrics.items()}
    
    return metrics, loss

# ********** 

def feature_extractor(model, data_loader, device = 'cuda:0', peri_flag = False):    
    emb = torch.tensor([])
    lbl = torch.tensor([], dtype = torch.int64)

    model = model.eval().to(device)
    
    with torch.no_grad():        
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

            emb = torch.cat((emb, x.detach().cpu()), 0)
            lbl = torch.cat((lbl, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Set Capacity\t: ', emb.size())
    assert(emb.size()[0] == lbl.size()[0])
    
    del data_loader
    time.sleep(0.0001)

    del model
    
    return emb, lbl


# **********