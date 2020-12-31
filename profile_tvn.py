import os
import sys
import torch
from torch import nn
import torchprof
from tqdm import tqdm
from tvn.model import TVN
from tvn.config import CFG1
from tvn.model import no_verbose


if __name__ == '__main__':
    no_verbose()
    torch.set_grad_enabled(False)
    tvn = TVN(CFG1, 52).cuda()
    x = torch.randn(23, 3, 224, 224).cuda()
    x_seq_lens = [12, 11]

    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        for _ in tqdm(range(100)):
            logits = tvn((x, x_seq_lens))

    print(prof.key_averages().table())
