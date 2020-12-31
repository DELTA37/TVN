import torch
from torch import nn
from torch.nn import functional as F
from tvn.cg.cg_block import ContextGatedConv2d


if __name__ == '__main__':
    x = torch.randn(1, 128, 64, 64)

    cg = ContextGatedConv2d(128, 256,
                            kernel_size=3,
                            stride=2,
                            padding=1)
    y = cg(x)

    print(x.shape)
    print(y.shape)
