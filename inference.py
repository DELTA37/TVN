import argparse
import os
import imageio
from tvn.model import TVN
from tvn.config import CFG1
from tvn.model import no_verbose


if __name__ == '__main__':
    no_verbose()
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('video_path')
    parser.add_argument('--num_classes', default=52, type=int)
    args = parser.parse_args()

    tvn = TVN(CFG1, args.num_classes).cuda()
