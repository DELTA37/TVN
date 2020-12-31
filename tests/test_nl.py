import torch
from torch import nn
from torch.nn import functional as F
from tvn.non_local.non_local_embedded_gaussian import NONLocalBlock2D


if __name__ == '__main__':
    x = torch.randn(1, 64, 128, 128)
    nl = NONLocalBlock2D(in_channels=64)
    y = nl(x)

    print(x.shape)
    print(y.shape)
