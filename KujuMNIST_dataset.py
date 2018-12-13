import os
import numpy as np
from torchvision.datasets.utils import makedir_exist_ok, download_url
from torch.utils.data import Dataset

class KujuMNIST_DS(Dataset):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz',
    ]
    base_filename = 'kmnist-{}-{}.npz'
    data_filepart = 'imgs'
    labels_filepart = 'labels'
    
    def __init__(self, folder, train_or_test='train', download=False, num_classes=10, max_items=None, tfms=None):
        self.root = os.path.expanduser(folder)
        if download:
            self.download()
            
        self.train = (train_or_test == 'train')
        
        self.data = np.load(os.path.join(self.root, self.base_filename.format(train_or_test, self.data_filepart)))    
        self.data = self.data['arr_0']
        self.targets = np.load(os.path.join(self.root, self.base_filename.format(train_or_test, self.labels_filepart)))    
        self.targets = self.targets['arr_0']
        self.c = num_classes
        self.max_items = max_items
        self.tfms = tfms
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        cur_data = np.expand_dims(self.data[index], axis=-1)

        if self.tfms:
            cur_data = self.tfms(cur_data)
        
        target = int(self.targets[index])
        img, target = cur_data, target
        
        return img, target

    def __len__(self):
        if self.max_items:
            return self.max_items
        return len(self.data)
    
    def download(self):
        makedir_exist_ok(self.root)
        for url in self.urls:
            filename = url.rpartition('/')[-1]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root, filename=filename, md5=None)