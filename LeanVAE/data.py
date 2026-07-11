import os
import os.path as osp
import math
import random
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import BatchSampler, Dataset, Sampler
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl
from typing import TypeVar, Optional, Iterator, List
from collections import Counter, defaultdict
from decord import VideoReader
from .utils.video_utils import VideoNorm, VideoNormWan
import pandas as pd

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def round_to_multiple(x, base=16):
    return int(round(x / base) * base)


class MultiSizeVideoDataset(data.Dataset):
    """CSV-driven video dataset used by the Wan-alignment branches.

    Reads video file paths from the ``videoID`` column of a CSV file, decodes a
    random contiguous clip of ``sequence_length`` frames, normalizes it to
    ``[-1, 1]`` via ``VideoNormWan``, and resizes to 480x832 to match the Wan
    VAE input resolution.
    """

    def __init__(self, data_list, data_folder=None, sequence_length=17, train=True, sample_rate=1, dynamic_sample=False):
        """
        Args:
            data_list (str): Path to a CSV file with a ``videoID`` column that
                holds the (absolute) path of each training video. Provided via
                the ``--train_datalist`` / ``--val_datalist`` CLI arguments.
            data_folder (Optional[str]): Unused. Kept for API compatibility.
            sequence_length (int): Number of frames per clip.
            sample_rate (int): Frame stride used when sampling.
            dynamic_sample (bool): If True, sample stride is drawn uniformly
                from ``[1, sample_rate]`` per clip.
        """
        super().__init__()
        self.train = train
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.dynamic_sample = dynamic_sample
        self.sample_rate = sample_rate

        df = pd.read_csv(data_list)
        self.annotations = df["videoID"].values.tolist()
        random.shuffle(self.annotations)
        self.norm = VideoNormWan()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        max_retries = getattr(self, "max_retries", 5)
        num_items = len(self.annotations)
        tried_indices = set()

        def try_load_video_by_index(index):
            video_path = self.annotations[index]
            try:
                decord_vr = VideoReader(video_path)
                total_frames = len(decord_vr)
            except Exception as e:
                raise RuntimeError(f"Failed to read video: {video_path}. Error: {e}")

            if self.dynamic_sample:
                sample_rate = random.randint(1, self.sample_rate)
            else:
                sample_rate = self.sample_rate

            required_frames = self.sequence_length * sample_rate

            if total_frames < self.sequence_length:
                raise RuntimeError(
                    f"Video {video_path} has only {total_frames} frames, "
                    f"but {self.sequence_length} are required."
                )

            if total_frames < required_frames:
                sample_rate = 1
                required_frames = self.sequence_length

            start_frame_ind = random.randint(0, max(0, total_frames - required_frames))
            end_frame_ind = min(start_frame_ind + required_frames, total_frames)
            frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, self.sequence_length, dtype=int
            )

            try:
                video_data = decord_vr.get_batch(frame_indice).asnumpy()
            except Exception as e:
                raise RuntimeError(f"Failed to decode frames from {video_path}. Error: {e}")

            video_data = torch.from_numpy(video_data).float()         # (T, H, W, C)
            video_data = video_data.permute(0, 3, 1, 2)               # (T, C, H, W)
            video = self.norm(video_data).permute(1, 0, 2, 3)         # (C, T, H, W)

            video = F.interpolate(video, size=(480, 832), mode="bilinear", align_corners=False)
            return {"video": video}

        # On failure, try a different random video before giving up.
        for attempt in range(max_retries):
            candidate_idx = idx if attempt == 0 else None

            if candidate_idx is None:
                available_indices = [i for i in range(num_items) if i not in tried_indices]
                if available_indices:
                    candidate_idx = random.choice(available_indices)
                else:
                    candidate_idx = random.randint(0, num_items - 1)

            tried_indices.add(candidate_idx)

            try:
                return try_load_video_by_index(candidate_idx)
            except Exception as e:
                print(f"[WARN] __getitem__ failed at index {candidate_idx}: {e}. Retrying another video...")

        # Fall back to a zero placeholder so training does not crash.
        C = 3
        placeholder = torch.zeros((C, self.sequence_length, 480, 832), dtype=torch.float32)
        return {"video": placeholder}


class VideoData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def _dataset(self, train):
        datasets = []
        for dataset_path, train_list, val_list in zip(self.args.data_path, self.args.train_datalist, self.args.val_datalist):
            dataset = MultiSizeVideoDataset(
                data_folder=dataset_path,
                data_list=train_list if train else val_list,
                sequence_length=self.args.sequence_length,
                train=train,
                sample_rate=self.args.sample_rate,
                dynamic_sample=self.args.dynamic_sample,
            )
            datasets.append(dataset)
        return datasets

    def _dataloader(self, train, steps=0, batch_size=None):
        dataset = self._dataset(train)
        if isinstance(self.args.batch_size, int):
            self.args.batch_size = [self.args.batch_size]
        self.batch_size = self.args.batch_size if batch_size is None else batch_size
        assert len(dataset) == len(self.args.batch_size)
        dataloaders = []
        for dset, d_batch_size in zip(dataset, self.batch_size):
            if dist.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                )
            else:
                sampler = None

            dataloader = data.DataLoader(
                dset,
                batch_size=d_batch_size,
                num_workers=self.args.num_workers if train else 0,
                pin_memory=False,
                sampler=sampler,
            )
            dataloaders.append(dataloader)

        return dataloaders

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)[0]

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, nargs="+", default=[''])
        parser.add_argument('--train_datalist', type=str, nargs="+", default=['./video/kinetics-dataset/train/datapath'])
        parser.add_argument('--val_datalist', type=str, nargs="+", default=['./video/kinetics-dataset/val/datapath'])

        parser.add_argument('--sequence_length', type=int, default=17)
        parser.add_argument('--sample_rate', type=int, default=1,
                            help='Frame sampling rate')
        parser.add_argument('--dynamic_sample', action='store_true',
                            help='Enable dynamic sampling rate')

        parser.add_argument('--batch_size', type=int, nargs="+", default=[5])
        parser.add_argument('--num_workers', type=int, default=40)
        return parser
