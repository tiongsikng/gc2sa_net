import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 42
save = True
write_log = True
method = 'GC2SA-Net'
remarks = 'Default_Settings'
dim = 1024

# Other hyperparameters
batch_sub = 4
batch_samp = 4
batch_size = batch_sub * batch_samp
random_batch_size = 16
test_batch_size = 100
epochs = 40
epochs_pre = 6
lr = 0.001
lr_sch = [6, 18, 30, 42]
w_decay = 1e-5
dropout = 0.1
momentum = 0.9

# Activate, or deactivate BatchNorm2D
# bn_flag = 0, 1, 2
bn_flag = 1
bn_moment = 0.1
if bn_flag == 1:
    bn_moment = 0.1

# Softmax Classifiers
af_s = 64
af_m = 0.35
cf_s = 64
cf_m = 0.35

# Activate / deactivate face_fc, peri_fc w.r.t. face_fc_ce_flag, peri_fc_ce_flag + network description
face_fc_ce_flag = True
peri_fc_ce_flag = True
face_peri_loss_flag = True

# CMB Loss Hyperparameters
tl_id = -1
tl_m = 1.0
tl_k = 1
tl_alpha = 10
tl_ap = 0.001 # 0.01++
if tl_id >= 0:
    tl_ap = 0.0

net_descr = remarks
net_tag = str('11_1')

bn_moment = float(bn_moment)
dropout = float(dropout)
af_s = float(af_s)
af_m = float(af_m)
cf_s = float(cf_s)
cf_m = float(cf_m)