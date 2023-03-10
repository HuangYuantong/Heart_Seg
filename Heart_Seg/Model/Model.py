from functools import partial

from Model.layers import AttentionBlock, N_Conv, DropBlock

import torch
import torch.nn as nn
from torch.nn import functional
from torchvision import models


def _new_forward_for_UNet_3Plus(self, x):
    # Encoder
    x_e1 = self.inc(x)  # [b,64,256,256]
    x_e2 = self.down1(x_e1)  # [b,256,128,128]
    x_e3 = self.down2(x_e2)
    x_e4 = self.down3(x_e3)
    x_d5 = self.down4(x_e4)
    # Decoder
    x_d4 = self.decoder4(x_e1, x_e2, x_e3, x_e4, x_d5)
    x_d3 = self.decoder3(x_e1, x_e2, x_e3, x_d4, x_d5)
    x_d2 = self.decoder2(x_e1, x_e2, x_d3, x_d4, x_d5)  # [b,320,128,128]
    x_d1 = self.decoder1(x_e1, x_d2, x_d3, x_d4, x_d5)  # [b,320,256,256]
    # output
    output = self.outc(x_d1)
    if self.deep_supervise:
        x_d3 = self.up3(self.outconv3(x_d3))
        return [output, x_d3], x_e2, x_d1  # 后两个用于分类网络的输入
    else:
        return [output], x_e2, x_d1  # 后两个用于分类网络的输入


class Model(nn.Module):
    def __init__(self, n_classification: int, unet_3plus: nn.Module,
                 dropout_p: float, dropout_size: int, close_drop=True):
        super(Model, self).__init__()
        self.n_classification = n_classification
        # 分割网络
        self.unet_3plus = unet_3plus
        # 重置分割网络的forward函数
        self.unet_3plus.forward = partial(_new_forward_for_UNet_3Plus, self.unet_3plus)
        # 冻结分割网络
        for i in self.unet_3plus.parameters(): i.requires_grad = False

        # # 使用Attention-UNet的注意力形式融合x_e2, x_d1
        self.inc_x_e2 = N_Conv(256, 256, n=1)  # [b,256,128,128]
        self.inc_x_d1 = nn.Sequential(N_Conv(320, 256, n=1),  # [b,256,128,128]，减小channel、下采样x2
                                      nn.AvgPool2d(2), )
        self.attention = AttentionBlock(256, 256, 256)  # [b,256,128,128]
        self.inc = nn.Sequential(DropBlock(dropout_p, dropout_size),
                                 N_Conv(256, 128, n=1),  # [b,128,64,64]，减小channel、下采样x2
                                 nn.AvgPool2d(2), ) if not close_drop \
            else nn.Sequential(N_Conv(256, 128, n=1), nn.AvgPool2d(2), )
        # 分类网络
        densenet = models.densenet121(False)
        self.net_classify = nn.Sequential(densenet.features.denseblock2, densenet.features.transition2,
                                          densenet.features.denseblock3, densenet.features.transition3,
                                          densenet.features.denseblock4, densenet.features.norm5, )
        self.classifier = nn.Sequential(densenet.classifier,
                                        nn.Dropout(0.05),
                                        nn.Linear(1000, self.n_classification), ) if not close_drop \
            else nn.Sequential(densenet.classifier, nn.Linear(1000, self.n_classification), )

    def forward(self, x):
        # 分割网络
        outputs_segment, x_e2, x_d1 = self.unet_3plus(x)  # 分割网络输出、x_e2、x_d1
        # 使用x_d1对x_e2进行attention
        x_e2 = x_e2 + self.inc_x_e2(x_e2)
        x_d1 = self.inc_x_d1(x_d1)
        attention_result = self.attention(x_d1, x_e2)  # 角色类似于：g蒙板，x输入，attention*x作为输出
        # 分类网络
        inputs = self.inc(attention_result)
        x = self.net_classify(inputs)  # [b, 1024, 16, 16]
        x = functional.relu(x, inplace=True)
        x = functional.adaptive_avg_pool2d(x, (1, 1))  # [b, 1024, 1, 1]
        x = torch.flatten(x, 1)  # [b, 1024]
        outputs_classify = self.classifier(x)
        return outputs_segment, outputs_classify
