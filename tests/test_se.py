import torch
from torch import nn
from torch.nn import functional as F
from tvn.se.se_block import SEBasicBlock


if __name__ == '__main__':
    x = torch.randn(1, 128, 32, 32)

    block = SEBasicBlock(128, 128, stride=2,
                         downsample=nn.AdaptiveAvgPool2d((16, 16)))
    y = block(x)

    print(x.shape)
    print(y.shape)
