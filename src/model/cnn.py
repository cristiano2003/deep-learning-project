import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self._make_layer(1, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = self._make_layer(128, 256)
        self.layer5 = self._make_layer(256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer6 = self._make_layer(256, 512)
        self.layer7 = self._make_layer(512, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8 = self._make_layer(512, 512)
        self.layer9 = self._make_layer(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 36)
        )

    def forward(self, x):
        x = self.layer1(x)   # B 64 112 112
        x = self.maxpool1(x)  # B 64 56 56
        x = self.layer2(x)   # B 128 56 56
        x = self.layer3(x)   # B 128 56 56
        x = self.maxpool2(x)  # B 128 28 28
        x = self.layer4(x)   # B 256 28 28
        x = self.layer5(x)   # B 256 28 28
        x = self.maxpool3(x)  # B 256 14 14
        x = self.layer6(x)   # B 512 14 14
        x = self.layer7(x)   # B 512 14 14
        x = self.maxpool4(x)  # B 512 7 7
        x = self.layer8(x)   # B 512 7 7
        x = self.layer9(x)   # B 512 7 7
        x = self.avgpool(x)  # B 512 1 1
        x = self.fc(x)       # B 36
        return x

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


if __name__ == "__main__":
    cnn = CNN()
    x = torch.randn(2, 1, 112, 112)
    y = cnn(x)
    print(y.shape)
    # calculate the number of parameters
    num_params = 0
    for param in cnn.parameters():
        num_params += param.numel()
    print(num_params)
