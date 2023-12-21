import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import os, sys
import math
import sys
import random
import numpy as np
from collections import deque
from itertools import combinations
import matplotlib.pyplot as plt

# *** *** *** *** ***

# ***** Modality and Augmentation Aware Contrastive Loss *****

class MACL(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda:0', intra_weighting = 1.0, inter_weighting = 1.25):
        super(MACL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device
        self.intra_weighting = intra_weighting
        self.inter_weighting = inter_weighting


    def forward(self, features, labels=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # convert labels into integers of shape (-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        #### Original SimCLR and SupConLoss masks ####
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        simclr_mask = torch.eye(batch_size, dtype=torch.bool).to(self.device).repeat(anchor_count, contrast_count)
        supcon_mask = torch.eq(labels, labels.T).float().to(self.device).repeat(anchor_count, contrast_count)
        
        #### Self Declared Positive and Negative Masks ####
        simclr_index = [0,1,3]
        negative_index = [1,3]
        positive_dict = {}
        negative_dict = {}
        intra_aug_dict = {}
        inter_aug_dict = {}

        for i in range(4):
            # for positive mask
            if i in simclr_index:
                positive_dict[i] = torch.eye(batch_size, dtype=torch.float32).to(self.device)
                inter_aug_dict[i] = torch.zeros_like(torch.eye(batch_size, dtype=torch.float32)).to(self.device)
                # negative_dict[i] = torch.zeros_like(positive_dict[i], dtype=torch.float32).to(device)
            else:
                positive_dict[i] = torch.eq(labels, labels.T).float().to(self.device)
                inter_aug_dict[i] = torch.ones_like(torch.eye(batch_size, dtype=torch.float32)).to(self.device)
                # negative_dict[i] = 1 - positive_dict[i]
            # for negative mask
            if i in negative_index:         
                negative_dict[i] = torch.zeros_like(positive_dict[i], dtype=torch.float32).to(self.device)
            else:
                negative_dict[i] = 1 - torch.eq(labels, labels.T).float().to(self.device)
            # for intra modal, intra augmentation mask
            if i == 0:
                intra_aug_dict[i] = torch.ones_like(torch.eye(batch_size, dtype=torch.float32)).to(self.device)
            else:
                intra_aug_dict[i] = torch.zeros_like(torch.eye(batch_size, dtype=torch.float32)).to(self.device)
        
        static_index_list = deque([0, 1, 2, 3])
        pos_mask = torch.Tensor(()).to(self.device)
        neg_mask = torch.Tensor(()).to(self.device)        
        intra_aug_mask = torch.Tensor(()).to(self.device)
        inter_aug_mask = torch.Tensor(()).to(self.device)
        
        for rows in range(len(static_index_list)):
            horizontal_pos = torch.Tensor(()).to(self.device)    
            horizontal_neg = torch.Tensor(()).to(self.device)
            horizontal_intra = torch.Tensor(()).to(self.device)
            horizontal_inter = torch.Tensor(()).to(self.device)
            for i in static_index_list:
                horizontal_pos = torch.cat((horizontal_pos, positive_dict[i]), dim = 0)
                horizontal_neg = torch.cat((horizontal_neg, negative_dict[i]), dim = 0)
                horizontal_intra = torch.cat((horizontal_intra, intra_aug_dict[i]), dim = 0)
                horizontal_inter = torch.cat((horizontal_inter, inter_aug_dict[i]), dim = 0)
            if rows == 0:
                pos_mask = horizontal_pos
                neg_mask = horizontal_neg
                intra_aug_mask = horizontal_intra
                inter_aug_mask = horizontal_inter
                # neg_mask = torch.cat((negative_dict[0], negative_dict[0], negative_dict[2], negative_dict[2]), dim=0)
            else:
                pos_mask = torch.cat((pos_mask, horizontal_pos), dim=1)                
                neg_mask = torch.cat((neg_mask, horizontal_neg), dim=1)
                intra_aug_mask = torch.cat((intra_aug_mask, horizontal_intra), dim=1)
                inter_aug_mask = torch.cat((inter_aug_mask, horizontal_inter), dim=1)             
            static_index_list.rotate(1)

        #### compute cosine similarity and logits ####
        cs_matrix = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # consider only negative sampled values and perform rescaling
        negative_logs = 4 * torch.tanh(cs_matrix) # perform new scaling on cosine similarity matrix
        negative_rescaled_log = torch.div(negative_logs, self.temperature) * neg_mask # rescale with temperature, only select negative

        # get positive and unconsidered negative (not sampled as negative) values (and remove negative sampled values)
        positive_nonnegative_logs = anchor_dot_contrast * (1 - neg_mask) # without rescale on positive

        # total of positive, unconsidered negatives, and considered (sampled) negatives
        anchor_dot_contrast = positive_nonnegative_logs + negative_rescaled_log # combine (positive + unconsidered negative) logs with (considered negative) logs

        # remove maximum after rescaling for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        ####
        threshold_mask = pos_mask.bool()
        final_mask = (threshold_mask | simclr_mask).float()

        positive_mask = final_mask * logits_mask # remove same instances, same modalities, but keep different augmentations

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        final_inter_aug_mask = (1 - inter_aug_mask) + (inter_aug_mask * self.inter_weighting)
        final_intra_aug_mask = (1 - intra_aug_mask) + (intra_aug_mask * self.intra_weighting)

        weighted_log_prob = log_prob * final_inter_aug_mask
        weighted_log_prob = weighted_log_prob * final_intra_aug_mask

        # compute mean of log-likelihood over positive        
        mean_log_prob_pos = (positive_mask * weighted_log_prob).sum(1) / final_mask.sum(1) # weighted logits (excluding trivial positives - diagonals)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def plot_mask(self, mask, img_name):
        plt.imshow(mask.detach().cpu().numpy()) 
        # plt.xticks(np.arange(1, int(mask.shape[0]), 1))
        plt.colorbar()
        plt.savefig(img_name, bbox_inches='tight')

# ****************   
        
# ***** CrossEntropy *****

class CrossEntropy(nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super(CrossEntropy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
   
    def forward(self, x):
        
        x = F.linear(F.normalize(x, p=2, dim=1), self.weight)
        # x = F.linear(F.normalize(x, p=2, dim=1), \
        #             F.normalize(self.weight, p=2, dim=1))

        return x
       
    def __repr__(self):

        return self.__class__.__name__ + '(' \
           + 'in_features = ' + str(self.in_features) \
           + ', out_features = ' + str(self.out_features) + ')'
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


# **************** 

# URL-1 : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# URL-2 : https://github.com/MuggleWang/CosFace_pytorch/blob/master/main.py
#    Args:
#        in_features: size of each input sample
#        out_features: size of each output sample
#        s: norm of input feature
#        m: margin

class CosFace(nn.Module):
    
    def __init__(self, in_features, out_features, s = 30.0, m = 0.40):
        
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def forward(self, input, label = None):
        
        # cosine = self.cosine_sim(input, self.weight).clamp(-1,1)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)
        
        return output# , F.normalize(self.weight, p=2, dim=1), (cosine * one_hot)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
    
    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

# ****************

# URL : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# Args:
#    in_features: size of each input sample
#    out_features: size of each output sample
#    s: norm of input feature
#    m: margin
#    cos(theta + m)

class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device = 'cuda:0'):
        
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # ***
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)) 
        self.reset_parameters() 
        self.device = device
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # nan issues: https://github.com/ronghuaiyang/arcface-pytorch/issues/32
        # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        
        return output #, self.weight[label,:]
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
    
# ****************
    
#### CMB Loss
    
def hardest_negative(loss_values, k):    
    '''
    hard_negative = np.argsort(loss_values)[::-1][k]
    return hard_negative if loss_values[hard_negative] > 0 else None
    '''
    hard_negative = np.argsort(loss_values)[::-1][:k]

    return hard_negative if len(hard_negative) > 0 else None


class TripletSelector:    
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError
    

def HardestNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative,cpu=cpu)


class FunctionNegativeTripletSelector(TripletSelector):    
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu = True):        
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels, label_min = 0, k = 1):
        
        if self.cpu:
            embeddings = embeddings.cpu()
        
        distance_matrix = 1 - torch.clamp(torch.mm(embeddings, embeddings.t()), min = -1.0, max = 1.0)
        distance_matrix = distance_matrix.cpu()
        
        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):            
            if label >= label_min:            
                label_mask = (labels == label)
                label_indices = np.where(label_mask)[0]
                if len(label_indices) < 2:
                    continue
                    
                negative_indices = np.where(np.logical_not(label_mask))[0]
                np.random.seed(int(torch.sum(distance_matrix))*label)
                label_indices = np.random.permutation(label_indices)

                anchor_positives = list(combinations(label_indices, 2))
                anchor_positives = np.array(anchor_positives)

                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):                    
                    an_distance = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])),\
                                                  torch.LongTensor(negative_indices)] 
                    loss_values = ap_distance - an_distance + self.margin                    
                    loss_values = loss_values.data.cpu().numpy()                    
                    hard_negative = self.negative_selection_fn(loss_values,k)
                    
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], int(hard_negative)])                    
                    
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])            
        triplets = np.array(triplets)
        
        return torch.LongTensor(triplets)
    

class OnlineTripletLoss(nn.Module):    
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """
    '''
        Usage: 'loss_cmb' : OnlineTripletLoss(tl_m, tl_ap, HardestNegativeTripletSelector(tl_m), device)
        
                tl_min = 0
                if net_params['tl_id'] == 2:
                    tl_min = net_params['face_num_sub']                    
                tl_k = net_params['tl_k']
                tl_ap_flag = False
                if net_params['tl_id'] < 0:
                    tl_ap_flag = True
                loss_cmb, ap, an = loss_fn['loss_cmb'](torch.cat((face_emb, peri_emb), dim = 0), \
                                                torch.cat((face_lbl, face_lbl)), \                                                
                                                tl_min, tl_k, tl_ap_flag)
    '''

    def __init__(self, margin = 0.7, ap_weight = 0.0, triplet_selector = None, device = 'cuda:0'):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.ap_weight = ap_weight
        self.triplet_selector = triplet_selector
        self.device = device
        
    def forward(self, embeddings, target, target_min = 0, k = 1, ap_flag = False):        
        triplets = self.triplet_selector.get_triplets(embeddings, target, target_min, k)
        
        triplets = triplets.to(self.device)
        
        ap_distances = torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]], 
                                                       embeddings[triplets[:, 1]].t())), min=-1., max=1.)

        an_distances = torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]], 
                                                       embeddings[triplets[:, 2]].t())), min=-1., max=1.)
        
        
        losses = torch.exp( ( an_distances - ap_distances ) / self.margin )
        
        ap = ( 1.0 - ap_distances ).mean().cpu()
        an = ( 1.0 - an_distances ).mean().cpu()

        losses_reg = 0.0

        if ap_flag is True:
            losses_reg = torch.sum( torch.exp( an_distances - ap_distances * self.margin ) ) * self.ap_weight
        
        losses = losses.mean() + losses_reg
        
        return losses, ap, an


# ****************
    
# Center Loss
class CrossModalCenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim=512, use_gpu=True, device = 'cuda:0'):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device = device

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0) 
        centers = F.normalize(self.centers, p=2, dim=1)
        distmat = 1 - torch.matmul(x, centers.t()).clamp(-1, 1)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

# ****************   
    
#### SupConLoss and SimCLR
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda:1'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # convert labels into integers of shape (-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# ****************   

#### LargeMargin and CPFC losses for FPCI

class LargeMarginSoftmax(nn.CrossEntropyLoss):
    """
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    regularization proposed in
       T. Kobayashi, "Large-Margin In Softmax Cross-Entropy Loss." In BMVC2019.
    """

    def __init__(self, reg_lambda=0.3, deg_logit=None,
                 weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginSoftmax, self).__init__(weight=weight, size_average=size_average,
                                                 ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0)  # number of samples
        C = input.size(1)  # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N), target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (C - 1)) * F.log_softmax(X, dim=1) * (1.0 - Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


class total_LargeMargin_CrossEntropy(nn.Module):
    def __init__(self):
        super(total_LargeMargin_CrossEntropy, self).__init__()
        self.loss1 = LargeMarginSoftmax()
        self.loss2 = LargeMarginSoftmax()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s1_loss = self.loss1(s1, target)
        s2_loss = self.loss2(s2, target)

        total_loss = s1_loss + s2_loss

        return total_loss


class CFPC_loss(nn.Module):
    """
    This combines the CrossCLR Loss proposed in
       M. Zolfaghari et al., "CrossCLR: Cross-modal Contrastive Learning For Multi-modal Video Representations,"
       In ICCV2021.
    """

    def __init__(self, temperature=0.02, negative_weight=0.8, device='cuda:0'):
        super(CFPC_loss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.device = device
        self.negative_w = negative_weight  # Weight of negative samples logits.

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask)
        return mask.to(self.device)

    def forward(self, face_features, ocular_features):
        """
        Inputs shape (batch, embed_dim)
        Args:
            face_features: face embeddings (batch, embed_dim)
            ocular_features: ocular embeddings (batch, embed_dim)
        Returns:
        """
        batch_size = face_features.shape[0]

        # Normalize features
        # face_features = nn.functional.normalize(face_features, dim=1)
        # ocular_features = nn.functional.normalize(ocular_features, dim=1)

        # Inter-modality alignment
        logits_per_face = face_features @ ocular_features.t()
        logits_per_ocular = ocular_features @ face_features.t()

        # Intra-modality alignment
        logits_clstr_face = face_features @ face_features.t()
        logits_clstr_ocular = ocular_features @ ocular_features.t()

        logits_per_face /= self.temperature
        logits_per_ocular /= self.temperature
        logits_clstr_face /= self.temperature
        logits_clstr_ocular /= self.temperature

        positive_mask = self._get_positive_mask(face_features.shape[0])
        negatives_face = logits_clstr_face * positive_mask
        negatives_ocular = logits_clstr_ocular * positive_mask

        face_logits = torch.cat([logits_per_face, self.negative_w * negatives_face], dim=1)
        ocular_logits = torch.cat([logits_per_ocular, self.negative_w * negatives_ocular], dim=1)

        diag = np.eye(batch_size)
        mask_face = torch.from_numpy(diag).to(self.device)
        mask_ocular = torch.from_numpy(diag).to(self.device)

        mask_neg_f = torch.zeros_like(negatives_face)
        mask_neg_o = torch.zeros_like(negatives_ocular)
        mask_f = torch.cat([mask_face, mask_neg_f], dim=1)
        mask_o = torch.cat([mask_ocular, mask_neg_o], dim=1)

        loss_f = self.compute_loss(face_logits, mask_f)
        loss_o = self.compute_loss(ocular_logits, mask_o)

        return (loss_f.mean() + loss_o.mean()) / 2