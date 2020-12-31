import torch
from torch import nn
from torch.nn import functional as F
from .temporal.temporal_blocks import TemporalAvgPool1d, TemporalMaxPool1d, TemporalConv1d, TemporalGlobalAvgPool
from .cg.cg_block import ContextGatedConv2d
from .se.se_block import SEBasicBlock
from .non_local.non_local_embedded_gaussian import NONLocalBlock2D
from collections import namedtuple


TBlock = namedtuple('TBlock', [
    'name',
    'in_channels',
    'out_channels',
    'spatial_ksize',
    'spatial_stride',
    'temporal_ksize',
    'temporal_stride',
    'temporal_pool_type',
    'cg_ksize',
    'cg_stride',
])
TBlock.__new__.__defaults__ = (None,) * len(TBlock._fields)
VERBOSE = True


def no_verbose():
    global VERBOSE
    VERBOSE = False


def verbose():
    global VERBOSE
    VERBOSE = True


def create_block(t_block: TBlock) -> nn.Module:
    if t_block.name == 'Block1':
        return Block1(in_channels=t_block.in_channels,
                      out_channels=t_block.out_channels,
                      spatial_ksize=t_block.spatial_ksize,
                      spatial_stride=t_block.spatial_stride,
                      temporal_ksize=t_block.temporal_ksize,
                      temporal_stride=t_block.temporal_stride,
                      temporal_pool_type=t_block.temporal_pool_type,
                      cg_ksize=t_block.cg_ksize,
                      cg_stride=t_block.cg_stride)
    elif t_block.name == 'Block2':
        return Block2(in_channels=t_block.in_channels,
                      out_channels=t_block.out_channels,
                      temporal_ksize=t_block.temporal_ksize,
                      temporal_stride=t_block.temporal_stride)
    elif t_block.name == 'Block3':
        return Block3(in_channels=t_block.in_channels,
                      spatial_ksize=t_block.spatial_ksize,
                      temporal_ksize=t_block.temporal_ksize)
    elif t_block.name == 'Block4':
        return Block4(in_channels=t_block.in_channels,
                      out_channels=t_block.out_channels,
                      temporal_ksize=t_block.temporal_ksize,
                      temporal_stride=t_block.temporal_stride,
                      cg_ksize=t_block.cg_ksize)
    else:
        raise NotImplementedError()


class Block1(nn.Module):
    def __init__(self, in_channels, out_channels,
                 spatial_ksize=3,
                 spatial_stride=1,
                 temporal_ksize=2,
                 temporal_stride=2,
                 temporal_pool_type='max',
                 cg_ksize=3,
                 cg_stride=2):
        super(Block1, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=spatial_ksize,
                                      stride=spatial_stride,
                                      padding=(spatial_ksize - 1) // 2)

        if temporal_pool_type == 'max':
            self.pool1d = TemporalMaxPool1d(temporal_ksize,
                                            stride=temporal_stride,
                                            padding=(temporal_ksize - 1) // 2)
        elif temporal_pool_type == 'avg':
            self.pool1d = TemporalAvgPool1d(temporal_ksize,
                                            stride=temporal_stride,
                                            padding=(temporal_ksize - 1) // 2)
        else:
            raise NotImplementedError()

        self.cg = ContextGatedConv2d(out_channels, out_channels,
                                     kernel_size=cg_ksize,
                                     stride=cg_stride,
                                     padding=(cg_ksize - 1) // 2)

    def forward(self, input):
        x, seq_lens = input
        x = self.spatial_conv(x)
        if VERBOSE:
            print('Block1 inputs', x.shape, seq_lens)
        x, seq_lens = self.pool1d(x, seq_lens)
        if VERBOSE:
            print('Block1 outputs', x.shape, seq_lens)
        x = self.cg(x)
        return x, seq_lens


class Block2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 temporal_ksize=3,
                 temporal_stride=2):
        super(Block2, self).__init__()

        self.conv1d = TemporalConv1d(in_channels, out_channels,
                                     kernel_size=temporal_ksize,
                                     stride=temporal_stride,
                                     padding=(temporal_ksize - 1) // 2)
        self.se = SEBasicBlock(out_channels, out_channels)

    def forward(self, input):
        x, seq_lens = input
        x = F.adaptive_avg_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        if VERBOSE:
            print('Block2 inputs', x.shape, seq_lens)
        x, seq_lens = self.conv1d(x, seq_lens)
        if VERBOSE:
            print('Block2 outputs', x.shape, seq_lens)
        x = self.se(x)
        return x, seq_lens


class Block3(nn.Module):
    def __init__(self, in_channels,
                 spatial_ksize=3,
                 temporal_ksize=3):
        super(Block3, self).__init__()

        self.spatial_conv2d = nn.Conv2d(in_channels, in_channels,
                                        kernel_size=spatial_ksize,
                                        stride=1,
                                        padding=(spatial_ksize - 1) // 2)

        self.conv1d = TemporalConv1d(in_channels, in_channels,
                                     kernel_size=temporal_ksize,
                                     stride=1,
                                     padding=(temporal_ksize - 1) // 2)

        self.nl = NONLocalBlock2D(in_channels)

    def forward(self, input):
        x, seq_lens = input
        residual_x = x
        x = self.spatial_conv2d(x)
        if VERBOSE:
            print('Block3 inputs', x.shape, seq_lens)
        x, seq_lens = self.conv1d(x, seq_lens)
        if VERBOSE:
            print('Block3 outputs', x.shape, seq_lens)
        x = self.nl(x)
        x = residual_x + x
        return x, seq_lens


class Block4(nn.Module):
    def __init__(self, in_channels, out_channels,
                 temporal_ksize=3,
                 temporal_stride=2,
                 cg_ksize=3):
        super(Block4, self).__init__()

        self.conv1d = TemporalConv1d(in_channels, out_channels,
                                     kernel_size=temporal_ksize,
                                     stride=temporal_stride,
                                     padding=(temporal_ksize - 1) // 3)

        self.cg = ContextGatedConv2d(out_channels, out_channels,
                                     kernel_size=cg_ksize,
                                     stride=1,
                                     padding=(cg_ksize - 1) // 2)

        self.se = SEBasicBlock(out_channels, out_channels)

    def forward(self, input):
        x, seq_lens = input
        x = F.adaptive_avg_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        if VERBOSE:
            print('Block4 inputs', x.shape, seq_lens)
        x, seq_lens = self.conv1d(x, seq_lens)
        if VERBOSE:
            print('Block4 outputs', x.shape, seq_lens)
        x = self.cg(x)
        x = self.se(x)
        return x, seq_lens


class TVN(nn.Module):
    def __init__(self, t_blocks,
                 num_classes,
                 prepare_seq=True):
        super(TVN, self).__init__()
        self.body = nn.Sequential(*[create_block(t_block) for t_block in t_blocks])
        self.global_pool = TemporalGlobalAvgPool(squeeze=True)
        self.head = nn.Sequential(nn.Linear(512, 512),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(512, num_classes))
        self.prepare_seq = prepare_seq

    def forward(self, input,
                extract_feas=False):
        """

        :param input: (x, seq_lens): [sum(seq_lens), H, W, C], idx: idx + seq_len - frames from unique video
        :param extract_feas
        :return:
        """
        if self.prepare_seq:
            bs, c, l, h, w = input.size()
            input = (input.permute(0, 2, 1, 3, 4).reshape(bs * l, c, h, w), [l for _ in range(bs)])

        x, seq_lens = input
        x, seq_lens = self.body((x, seq_lens))
        feas, seq_lens = self.global_pool(x, seq_lens)
        if VERBOSE:
            print(seq_lens)
        assert all([seq_len == 1 for seq_len in seq_lens])
        if extract_feas:
            return feas
        logits = self.head(feas)
        return logits
