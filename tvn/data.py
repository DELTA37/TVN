import os
import sys
import json
import torchvision
from .data_augmentor import RandomCropVideo, RandomHorizontalFlipVideo, RandomReverseTimeVideo, RandomRotationVideo, IdentityTransform, Scale, ComposeMix
from torch.utils.data import Dataset
from .data_loader_skvideo import VideoFolder


class SomethingSomethingV2(VideoFolder):
    def __init__(self, root, mode='train'):
        upscale_size = int(84 * 1.1)
        assert mode in ['train', 'test', 'validation']
        labels_path = os.path.join(root, 'something-something-v2-labels.json')
        mode_path = os.path.join(root, f'something-something-v2-{mode}.json')
        videos_path = os.path.join(root, '20bn-something-something-v2')

        transform_pre = ComposeMix([
            [RandomRotationVideo(20), "vid"],
            [Scale(upscale_size), "img"],
            [RandomCropVideo(84), "vid"],
            [RandomHorizontalFlipVideo(0), "vid"],
            # [RandomReverseTimeVideo(1), "vid"],
            # [torchvision.transforms.ToTensor(), "img"],
        ])
        # identity transform
        transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
        ])
        super(SomethingSomethingV2, self).__init__(root=videos_path,
                                                   json_file_input=mode_path,
                                                   json_file_labels=labels_path,
                                                   is_val=(mode != 'train'),
                                                   transform_post=transform_post,
                                                   transform_pre=transform_pre,
                                                   clip_size=36,
                                                   nclips=1,
                                                   step_size=1)

    def __getitem__(self, index):
        data = super(SomethingSomethingV2, self).__getitem__(index)
        video, label = data
        return data
