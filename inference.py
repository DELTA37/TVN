import argparse
import os
import torch
import torchvision
from tvn.data_augmentor import ComposeMix
import imageio
import re
import sh
from tvn.model import TVN
from tvn.config import CFG1
from tvn.model import no_verbose
from skvideo.io import FFmpegReader
ffprobe = sh.ffprobe.bake('-v', 'error', '-show_entries',
                          'format=start_time,duration')


def get_duration(file):
    cmd_output = ffprobe(file)
    start_time, duration = re.findall("\d+\.\d+", str(cmd_output.stdout))
    return int(float(duration) - float(start_time))


if __name__ == '__main__':
    no_verbose()
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--num_classes', default=52, type=int)
    args = parser.parse_args()

    tvn = TVN(CFG1, args.num_classes)
    if args.ckpt_path is not None:
        tvn.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    tvn.cuda()

    reader = FFmpegReader(args.video_path, inputdict={}, outputdict={'-r': "12",
                                                                     '-vframes': str(12 * get_duration(args.video_path))})

    try:
        imgs = []
        for img in reader.nextFrame():
            imgs.append(img)
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, args.video_path))

    transform_post = ComposeMix([
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.CenterCrop(84), "img"],
    ])
    video = torch.stack(transform_post(imgs), dim=0).permute(1, 0, 2, 3).unsqueeze(0).cuda()
    print(video.shape)
    logits = tvn(video)
    print(logits.shape)
