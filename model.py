import os
import sys
sys.path.append(os.getcwd())
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from statistic import RunningMean

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self._make_layer(1,64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.layer2 = self._make_layer(64,128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.layer3 = self._make_layer(128,256)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,26)
        )

    def forward(self,x):
        x = self.layer1(x)   # B 64 28 28
        x = self.maxpool1(x) # B 64 14 14
        x = self.layer2(x)   # B 128 14 14
        x = self.maxpool2(x) # B 128 7 7
        x = self.layer3(x)   # B 256 7 7
        x = self.avgpool(x)  # B 256 1 1
        x = self.fc(x)       # B 26
        return x

    def _make_layer(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.bn1   = nn.BatchNorm2d(64)
        self.model.fc = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,26)
        )

    def forward(self,x):
        x = self.model.conv1(x)   # B 64 14 14
        x = self.model.bn1(x)     # B 64 14 14
        x = self.model.relu(x)    # B 64 14 14
        x = self.model.maxpool(x) # B 64 7 7
        x = self.model.layer1(x)  # B 64 7 7
        x = self.model.layer2(x)  # B 128 4 4
        x = self.model.layer3(x)  # B 256 2 2
        x = self.model.avgpool(x) # B 256 1 1
        x = torch.flatten(x, 1)   # B 256
        x = self.model.fc(x)      # B 26
        return x
    
class SignMNISTModel(pl.LightningModule):
    def __init__(self,model="resnet",lr=2e-4):
        super().__init__()
        if model == "resnet":
            self.model = ResNet()
        elif model == "cnn":
            self.model = CNN()

        self.train_loss = RunningMean()
        self.val_loss   = RunningMean()
        self.train_acc  = RunningMean()
        self.val_acc    = RunningMean()

        self.loss = nn.CrossEntropyLoss()
        self.lr   = lr

    def forward(self,x):
        return self.model(x)
    
    def _cal_loss_and_acc(self,batch):
        x,y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        return loss,acc
    
    def training_step(self,batch,batch_idx):
        loss,acc = self._cal_loss_and_acc(batch)
        self.train_loss.update(loss.item(),batch[0].shape[0])
        self.train_acc.update(acc.item(),batch[0].shape[0])
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss,acc = self._cal_loss_and_acc(batch)
        self.val_loss.update(loss.item(),batch[0].shape[0])
        self.val_acc.update(acc.item(),batch[0].shape[0])
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_loss",self.train_loss(),sync_dist=True)
        self.log("train_acc",self.train_acc(),sync_dist=True)
        self.train_loss.reset()
        self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        self.log("val_loss",self.val_loss(),sync_dist=True)
        self.log("val_acc",self.val_acc(),sync_dist=True)
        self.val_loss.reset()
        self.val_acc.reset()
    
    def test_step(self,batch,batch_idx):
        loss,acc = self._cal_loss_and_acc(batch)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

if __name__ == "__main__":
    res = ResNet()
    cnn = CNN()
    print(f"ResNet Output Shape: {res(torch.randn(4,1,28,28)).shape}")
    print(f"CNN Output Shape: {cnn(torch.randn(4,1,28,28)).shape}")