from UNet import UNet, Down, DoubleConv

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
        self.net2 = nn.Sequential(nn.Conv2d(self.clsfct_input_size, 1024 * 2, (1, 1)),
                                  Down(1024 * 2, 1024 * 2),  # [1024*4, 8, 8]
                                  Down(1024 * 2, 1024),  # [1024*4, 4, 4]
                                  Down(1024, 1024),  # [1024*2, 2, 2]
                                  nn.Conv2d(1024, 1024, (2, 2), stride=(2, 2))  # [1024, 1, 1]
                                  )
        self.net3 = nn.Sequential(nn.Linear(1024, 3),  # FC层
                                  )

    def forward(self, x):
        logits, bottom = self.UNet(x)
        # 使用双线性插值下（或上）采样
        inputs = torch.vstack([i.squeeze() for i in bottom])  # [n*1024, 16, 16]
        inputs = torch.permute(inputs, (1, 2, 0))  # [16, 16, n*1024]
        # [16, 16, clsfct_input_size]
        inputs = nn.functional.interpolate(inputs, size=self.clsfct_input_size, mode='linear', align_corners=True)
        inputs = torch.permute(inputs, (2, 0, 1)).unsqueeze(0)  # [1, clsfct_input_size, 16, 16]
        # 分类网络
        outputs = self.net2(inputs).squeeze(2, 3)  # [1, 1024]
        outputs = self.net3(outputs)  # [1,3]
        return logits, outputs
