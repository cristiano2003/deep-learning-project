import torch
import torch.nn as nn
from torchvision import models
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.model.bn1   = nn.BatchNorm2d(64)
        self.model.fc = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,36)
        )

    def forward(self,x):
        x = self.model.conv1(x)   # B 64 50 50
        x = self.model.bn1(x)     # B 64 50 50
        x = self.model.relu(x)    # B 64 50 50
        x = self.model.maxpool(x) # B 64 25 25
        x = self.model.layer1(x)  # B 64 25 25
        x = self.model.layer2(x)  # B 128 13 13
        x = self.model.layer3(x)  # B 256 7 7
        x = self.model.avgpool(x) # B 256 1 1
        x = torch.flatten(x, 1)   # B 256
        x = self.model.fc(x)      # B 26
        return x
    
if __name__ == "__main__":
    res = ResNet()
    x = torch.randn(2,1,112,112)
    y = res(x)
    print(y.shape)