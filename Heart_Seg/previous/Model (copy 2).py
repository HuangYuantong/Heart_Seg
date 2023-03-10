from UNet import UNet, Down, DoubleConv
from init_weights import init_weights

import torch
from torch import nn
from torch.nn import functional


class Model(nn.Module):
    def __init__(self, n_channels, n_classes, n_classification, bilinear=False):
        super(Model, self).__init__()
        self.n_classes = n_classes  # segmentation分类 (4)
        self.n_classification = n_classification  # n分类 (3)
        self.clsfct_input_size = 1024 * 5  # n分类时网络输入channel数
        # 分割网络
        self.UNet = UNet(n_channels, n_classes, bilinear)
        # 分类网络
        self.net2 = nn.Sequential(Down(1024, 1024),  # [1024, 8, 8]
                                  YoloBlock(1024, 512),  # [512, 8, 8]
                                  Down(512, 1024),  # [1024, 4, 4]
                                  nn.AdaptiveAvgPool2d((1, 1)),  # [1024, 1, 1]
                                  )
        self.net3 = nn.Sequential(nn.Linear(self.clsfct_input_size, 1000),
                                  nn.ReLU(),
                                  nn.Linear(1000, 3),
                                  )

    def forward(self, x):
        logits, bottom = self.UNet(x)
        inputs = self.net2(bottom)  # [n, 1024, 1, 1]
        # 使用双线性插值下（或上）采样
        inputs = torch.hstack([i.squeeze() for i in inputs]).unsqueeze(0).unsqueeze(0)  # [1, 1, 1024*n]
        # [1, clsfct_input_size]
        inputs = nn.functional.interpolate(inputs, size=self.clsfct_input_size, mode='linear', align_corners=True)
        # 分类网络
        outputs = self.net3(inputs.squeeze(0))  # [1,3]
        return logits, outputs


class YoloBlock(nn.Module):
    """(1x1 conv(out_c) => [BN] => ReLU => 3x3 conv(in_c)) * 2 => 1x1 conv(out_c)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self._double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, (3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.yolo_block = nn.Sequential(self._double_conv,
                                        self._double_conv,
                                        nn.Conv2d(in_channels, out_channels, (1, 1)))
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.yolo_block(x)
