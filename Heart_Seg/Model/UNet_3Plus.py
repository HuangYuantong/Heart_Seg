import torch.nn as nn

from Model.layers import N_Conv, OutConv, Decoder
from torchvision import models


class UNet_3Plus(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervise: bool = False):
        super(UNet_3Plus, self).__init__()
        self.n_classes = n_classes
        self.deep_supervise = deep_supervise
        filters = [64, 256, 512, 1024, 2048]
        catChannel = filters[0]
        upChannel = catChannel * 5
        # Encoder
        resnet = models.resnet50(False)
        self.inc = N_Conv(n_channels, filters[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2),
                                   resnet.layer1, )
        self.down2 = resnet.layer2
        self.down3 = resnet.layer3
        self.down4 = resnet.layer4
        # DeCoder（每个DeCoder = fusion(4个Jump Connection、1个up)）
        self.decoder4 = Decoder(4, filters, catChannel, upChannel)
        self.decoder3 = Decoder(3, filters, catChannel, upChannel)
        self.decoder2 = Decoder(2, filters, catChannel, upChannel)
        self.decoder1 = Decoder(1, filters, catChannel, upChannel)
        # output
        self.outc = OutConv(upChannel, n_classes)
        # Deep Supervise
        self.outconv3 = OutConv(upChannel, self.n_classes)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.outconv2 = OutConv(upChannel, self.n_classes)
        # self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x_e1 = self.inc(x)
        x_e2 = self.down1(x_e1)
        x_e3 = self.down2(x_e2)
        x_e4 = self.down3(x_e3)
        x_d5 = self.down4(x_e4)
        # Decoder
        x_d4 = self.decoder4(x_e1, x_e2, x_e3, x_e4, x_d5)
        x_d3 = self.decoder3(x_e1, x_e2, x_e3, x_d4, x_d5)
        x_d2 = self.decoder2(x_e1, x_e2, x_d3, x_d4, x_d5)
        x_d1 = self.decoder1(x_e1, x_d2, x_d3, x_d4, x_d5)
        # output
        output = self.outc(x_d1)
        if self.deep_supervise:
            # 如果需要进行深监督则进行上采样和输出
            x_d3 = self.up3(self.outconv3(x_d3))
            # x_d2 = self.up2(self.outconv2(x_d2))
            return [output, x_d3]
        else:
            return [output]


def weight_init(m):
    """根据网络层的不同定义不同的初始化方式"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
