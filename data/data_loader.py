import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import random
import cv2
from torch.nn import functional as F
from torch.utils.data.sampler import BatchSampler
from configs.params import batch_sub, batch_samp, seed, device, random_batch_size, test_batch_size

#### Data loader for network ####
device = device
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# print('Running on device: {}'.format(device))


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class BalancedBatchSampler(BatchSampler):
    
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = torch.tensor(labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # print(self.label_to_indices)
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):        
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # seed is used to enable same batches for both streams
            np.random.seed(self.count)
            random.seed(self.count)
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            # classes = np.random.choice(self.labels_set, self.n_classes, replace = True)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def ConvertRGB2BGR(x):
    x = np.float32(x)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR) 
    return x    


def FixedImageStandard(x):
    x = (x - 127.5) * 0.0078125
    return x


def gen_data(path_dir, mode, type='periocular', aug='False'):
    if mode == 'test' and aug == 'True':
        raise('Testing dataset has augmentation!')
    if type == 'face':
        sz = (112, 112)
    elif type == 'periocular' or type == 'peri':
        sz = (112, 112)
    
    data_trans = transforms.Compose( [ transforms.Resize(sz),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                    ] )
    aug_trans = transforms.Compose( [ transforms.RandomAffine(degrees=10, translate=None, scale=(1.0,1.2),
                                                                    shear=0),
                                    transforms.Resize(sz), 
                                    transforms.RandomHorizontalFlip(p=0.8), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                    ] )
        
    data_set = datasets.ImageFolder(path_dir, transform = data_trans)        
    data_sampler = BalancedBatchSampler(data_set.targets, n_classes = batch_sub, n_samples = batch_samp)
    if aug == 'True':
        data_set_aug = datasets.ImageFolder(path_dir, transform = aug_trans)
        # data_sets = data_set_aug
        data_sets = ConcatDataset(data_set, data_set_aug)
    else:
        data_sets = data_set

    if mode == 'train':
        data_loader = torch.utils.data.DataLoader(data_sets, batch_sampler = data_sampler, num_workers = 4,
                                              worker_init_fn = random.seed(seed))
    elif mode == 'train_rand':
        data_loader = torch.utils.data.DataLoader(data_sets, batch_size = random_batch_size, num_workers = 4,
                                              worker_init_fn = random.seed(seed), shuffle = True, drop_last = True)
    elif mode == 'test' and aug == 'False':
        data_loader = torch.utils.data.DataLoader( data_sets, batch_size = test_batch_size*4, shuffle = False, 
                                                num_workers = 6, worker_init_fn = random.seed(seed))
    
    return data_loader, data_set