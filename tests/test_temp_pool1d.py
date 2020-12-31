import os
import sys
from tvn.temporal.temporal_blocks import TemporalMaxPool1d
import torch


if __name__ == '__main__':
    x_seq_lens = [2, 5, 8, 3]
    x = torch.randn(sum(x_seq_lens), 128, 32, 32).cuda()

    pool = TemporalMaxPool1d(kernel_size=2,
                             stride=2)
    pool = pool.cuda()

    y, y_seq_lens = pool(x, x_seq_lens)
    print(x.shape, x_seq_lens)
    print(y.shape, y_seq_lens)
