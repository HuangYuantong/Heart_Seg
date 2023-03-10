import torch
from torch import nn
from torch.nn import functional


class N_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * n"""

    def __init__(self, in_size, out_size, n=2, ks=3, stride=1, padding=1, is_batchnorm=True):
        super(N_Conv, self).__init__()
        self.n = n
        for i in range(self.n):
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, stride, padding, bias=False),
                                 nn.BatchNorm2d(out_size),
                                 nn.ReLU(inplace=True),
                                 ) if is_batchnorm else \
                nn.Sequential(nn.Conv2d(in_size, out_size, ks, stride, padding, bias=False),
                              nn.ReLU(inplace=True), )
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(self.n):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            N_Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = N_Conv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = N_Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.sizes()[2] - x1.sizes()[2]
        diffX = x2.sizes()[3] - x1.sizes()[3]
        # pad(input, pad=[左, 右, 上, 下, 前, 后]所需填充范围, mode填充模式)
        x1 = functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_decoder, filters, catChannel, upChannel):
        super(Decoder, self).__init__()
        self.n_decoder = n_decoder
        down_scale = [1, 2, 4, 8, 16]
        # 下采样：n_decoder-1个，平连：1，上采样：5-n_decoder个
        # 共4+1个输入（其中1个是上一级DeCoder），通道数都为catChannel（64），仅来自第五层的为filters[4]
        for i in range(5):
            # 确定上、下采样倍率差
            scale = int(down_scale[n_decoder - 1] / down_scale[i])
            re_scale = int(down_scale[i] / down_scale[n_decoder - 1])
            # 根据倍率差决定下采样、平接、上采样
            conv = nn.Sequential(nn.MaxPool2d(int(scale), ceil_mode=True),
                                 N_Conv(filters[i], catChannel, 1), ) if scale > 1 \
                else N_Conv(filters[i], catChannel, 1) if scale == 1 \
                else nn.Sequential(nn.Upsample(scale_factor=int(re_scale), mode='bilinear', align_corners=True),
                                   N_Conv(filters[i] if i == 4 else upChannel, catChannel, 1), )
            setattr(self, f'connection_{i + 1}', conv)
        # 一个单层将5个输入融合
        self.fusion = N_Conv(upChannel, upChannel, 1)

    def forward(self, x_layer1, x_layer2, x_layer3, x_layer4, x_layer5):
        x1 = self.connection_1(x_layer1)
        x2 = self.connection_2(x_layer2)
        x3 = self.connection_3(x_layer3)
        x4 = self.connection_4(x_layer4)
        x5 = self.connection_5(x_layer5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.fusion(x)
        return x


class AttentionBlock(nn.Module):
    """Attention Gate"""

    def __init__(self, in_channels_g, in_channels_x, out_channels):
        super(AttentionBlock, self).__init__()
        # 1x1卷积 => 键值
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))
        # 相加、relu、1x1卷积、Sigmoid => [b,1,w,h]权重
        self.psi = nn.Sequential(
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(), )

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.psi(g + x)
        return x * psi


class DropBlock(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        super(DropBlock, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self.drop_prob / (self.block_size ** 2)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = functional.max_pool2d(input=mask[:, None, :, :],
                                           kernel_size=(self.block_size, self.block_size),
                                           stride=(1, 1),
                                           padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask
