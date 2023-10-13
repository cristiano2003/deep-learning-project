import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
import numpy as np

class SignMNISTDataset(Dataset):
    def __init__(self,type='train'): # train, test
        if type == 'train':
            self.transform = T.Compose([
                T.ToPILImage('L'),
                T.RandomChoice([T.RandomRotation(degrees=15)],p=[0.3]),
                T.ToTensor()
            ])
        else:
            self.transform = None

        df = pd.read_csv(os.path.join(os.getcwd(),f"data/sign_mnist_{type}/sign_mnist_{type}.csv"))
        self.X = np.array(df.drop(columns=['label']), dtype=np.float32)
        self.X = self.X.reshape(-1,1,28,28)
        self.Y = np.array(df['label'])
        self.Y = np.expand_dims(self.Y,axis=1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(torch.from_numpy(self.X[index])),\
                   torch.from_numpy(self.Y[index])
        else:
            return torch.from_numpy(self.X[index]),\
                   torch.from_numpy(self.Y[index])