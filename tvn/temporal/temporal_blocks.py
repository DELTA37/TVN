import torch
from torch import nn
from torch.nn import functional as F


class Temporal(nn.Module):
    def __init__(self):
        super(Temporal, self).__init__()

    def forward_single(self, x: torch.Tensor):
        raise NotImplementedError()

    def forward(self, x, seq_lens):
        out = []
        new_seq_lens = []
        for i in range(len(seq_lens)):
            idx = sum(seq_lens[:i])
            length = seq_lens[i]

            if len(seq_lens) == 1:
                """
                Option for batch_size = 1
                """
                temp_x = x
            else:
                temp_x = x[idx: idx + length]

            y = self.forward_single(temp_x)

            new_seq_len = y.size(0)
            new_seq_lens.append(new_seq_len)

            out.append(y)

        if len(seq_lens) == 1:
            """
            Option for batch_size = 1
            """
            return out[0], new_seq_lens
        else:
            return torch.cat(out, dim=0), new_seq_lens


class TemporalMaxPool1d(Temporal):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0):
        super(TemporalMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward_single(self, x: torch.Tensor):
        seq_len, c, h, w = x.size()
        x = torch.transpose(x.view(seq_len, c, h * w), 2, 0)  # [bs, channels, L]
        y = F.max_pool1d(x,
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding=self.padding)
        y = torch.transpose(y, 2, 0).view(y.size(2), y.size(1), h, w)
        return y


class TemporalAvgPool1d(Temporal):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0):
        super(TemporalAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward_single(self, x: torch.Tensor):
        seq_len, c, h, w = x.size()
        x = torch.transpose(x.view(seq_len, c, h * w), 2, 0)  # [bs, channels, L]
        y = F.avg_pool1d(x,
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding=self.padding)
        y = torch.transpose(y, 2, 0).view(y.size(2), y.size(1), h, w)
        return y


class TemporalConv1d(Temporal):
    def __init__(self,
                 in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(TemporalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups)

    def forward_single(self, x: torch.Tensor):
        seq_len, c, h, w = x.size()
        y = torch.transpose(x.view(seq_len, c, h * w), 2, 0)  # [bs, channels, L]
        y = self.conv(y)
        y = torch.transpose(y, 2, 0).view(y.size(2), y.size(1), h, w)
        return y


class TemporalGlobalAvgPool(Temporal):
    def __init__(self, squeeze=True):
        super(TemporalGlobalAvgPool, self).__init__()
        self.squeeze = squeeze

    def forward_single(self, x: torch.Tensor):
        x = torch.mean(x, dim=[0, 2, 3], keepdim=True)
        if self.squeeze:
            x = x.squeeze(-1).squeeze(-1)
        return x
