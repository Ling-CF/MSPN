import torch
import torch.nn as nn
from NetBlock import Bottleneck


class discriminator(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256, 512, 1024), in_channel=3, size=(4,5)):
        super(discriminator, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_channel, out_channels=channels[0], kernel_size=(7,7), padding=3)
        necks = []
        for l in range(len(channels)-1):
            neck = Bottleneck(in_channels=channels[l], out_channels=channels[l+1], batchnorm=True)
            pool = nn.MaxPool2d(kernel_size=2)
            necks.append(neck)
            necks.append(pool)
        self.conv = nn.Sequential(*necks)
        self.full_conv = nn.Conv2d(in_channels=channels[-1], out_channels=1024, kernel_size=size)
        self.linear = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = (x-x.mean())/torch.std(x)
        batchs = x.size(0)
        x = self.layer1(x)
        x = self.conv(x)
        x = self.full_conv(x)
        x = x.reshape(batchs,-1)
        x = self.linear(x)
        return x