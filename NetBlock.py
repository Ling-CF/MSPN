import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, batchnorm=False):
        super(Bottleneck,self).__init__()
        # The construction method is similar to the ResNet
        if batchnorm:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=(3,3), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3),  padding=1),
                nn.BatchNorm2d(out_channels)
            )
            self.res = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            )
            self.res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.res(x)
        out = self.bottleneck(x)
        out += residual
        out = self.relu(out)
        return out



class EnDeNet(nn.Module):
    def __init__(self, channels, in_ch, out_ch, higher_channel, BN=False):
        super(EnDeNet, self).__init__()
        self.n_layers = len(channels) - 1
        self.out_conv = nn.Sequential(nn.Conv2d(2*channels[0], out_ch, kernel_size=(3,3), padding=1))
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=channels[0], kernel_size=(5,5), padding=2),
            nn.LeakyReLU(inplace=True)
        )
        if higher_channel != 0:
            self.Merge_Hidden = nn.Sequential(
                nn.Conv2d(in_channels=channels[-1]+higher_channel, out_channels=channels[-1], kernel_size=(3,3), padding=1, bias=False),
                nn.LeakyReLU(inplace=True)
            )
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float, requires_grad=True))
        # reverse the channels for the decoder
        reverse_ch = sorted(channels, reverse=True)

        # setting encoder layer and decoder layer for each network level respectively
        for l in range(self.n_layers):
            encoder = nn.Sequential(
                Bottleneck(in_channels=channels[l], out_channels=channels[l+1], batchnorm=BN),
                nn.MaxPool2d(kernel_size=2),
                Bottleneck(in_channels=channels[l+1], out_channels=channels[l+1], batchnorm=BN)
            )
            setattr(self, 'encoder{}'.format(l), encoder)
            upsample = nn.Sequential(
                Bottleneck(in_channels=reverse_ch[l], out_channels=reverse_ch[l+1], batchnorm=BN),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Bottleneck(in_channels=reverse_ch[l+1], out_channels=reverse_ch[l+1], batchnorm=BN),
            )
            setattr(self, 'upsample{}'.format(l), upsample)
            merge_conv = nn.Sequential(
                nn.Conv2d(2*reverse_ch[l+1], reverse_ch[l+1], kernel_size=(3,3), padding=1),
                nn.ReLU(inplace=True)
            )
            setattr(self, 'merge_conv{}'.format(l), merge_conv)


    def forward(self, x,  higher_h):
        feature_maps = []  # Used to save the feature map generated by the encoder

        # encoding
        x = self.in_conv(x)
        feature_maps.append(x)
        for l in range(self.n_layers):
            encoder = getattr(self, 'encoder{}'.format(l))
            x = encoder(x)
            feature_maps.append(x)

        # decoding
        feature_maps.reverse()
        x = feature_maps[0]


        if higher_h != None:
            x = torch.cat([x, higher_h], dim=1)
            x = self.Merge_Hidden(x)

        current_h = x

        # print(x.size())
        for l in range(len(feature_maps)-1):
            upsample = getattr(self, 'upsample{}'.format(l))
            merge_conv = getattr(self, 'merge_conv{}'.format(l))
            up = upsample(x)
            merge = torch.cat([up, feature_maps[l+1]], dim=1)
            if l < len(feature_maps)-2:
                x = merge_conv(merge)
            else:
                x = self.out_conv(merge)
        return x, current_h


if __name__ == '__main__':
    model = EnDeNet(channels=(32,64,128,256,512), in_ch=6, out_ch=12, higher_channel=256)
    x1 = torch.rand(1,6,64,80)
    out = model(x1, None, None)
    print(out.size())