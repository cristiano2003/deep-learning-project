import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
import numpy as np
from PIL import Image
class ASLDataset(Dataset):
    def __init__(self,type='train'): # train, test
        if type == 'train':
            self.transform = T.Compose([
                T.Resize((112,112)),
                T.RandomChoice([T.RandomRotation(degrees=10)],p=[0.3]),
                T.ToTensor()
            ])
        else:
            self.transform = self.transform = T.Compose([
                T.Resize((112,112)),
                T.ToTensor()
            ])

        self.file_list = []
        labels = []
        for dirname, _, filenames in os.walk('asl_dataset'):
            for filename in filenames:
                self.file_list.append(os.path.join(dirname, filename))
                labels.append(dirname.split('/')[-1])
        dict_labels = {k: v for v, k in enumerate(sorted(set(labels)))}
        self.targets = [dict_labels[label] for label in labels]

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img = Image.open(self.file_list[index]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.targets[index]