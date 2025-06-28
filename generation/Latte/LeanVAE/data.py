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
from .utils.video_utils import VideoNorm

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
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

class MultiSizeVideoDataset(data.Dataset):
    """ A flexible dataset for loading videos of different resolutions stored in a structured format.
    This dataset reads video file paths from text files, where each file corresponds to a specific resolution (e.g., `256x256`).
    Returns BCTHW videos in the range [-0.5, 0.5] """
    def __init__(self, data_list, data_folder=None, sequence_length=17, train=True, sample_rate=1, dynamic_sample=False):
        """
        Args:
            data_list (str): Path to the folder containing text files with video paths.
            data_folder (Optional[str]): Root folder where videos are stored (if paths in data_list are relative).
               
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.dynamic_sample = dynamic_sample
        self.sample_rate = sample_rate

        lengths = []
        annotations = []
        for dir in os.listdir(data_list):
            file_path = os.path.join(data_list, dir)
            with open(file_path) as f:
                annotation = [ann.strip() for ann in f.readlines()]
                annotations.extend(annotation)
                lengths.extend([dir] * len(annotation))
                
        self.annotations = annotations
        self.lengths = lengths

        self.norm = VideoNorm()

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, idx):
        
        video_path = self.annotations[idx] if self.data_folder is None else os.path.join(self.data_folder, self.annotations[idx])
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
            raise RuntimeError(f"Video {video_path} has only {total_frames} frames, but {self.sequence_length} are required.")
        
        if total_frames < required_frames:
            sample_rate = 1
            required_frames = self.sequence_length
        
        start_frame_ind = random.randint(0, max(0, total_frames - required_frames))
        end_frame_ind = min(start_frame_ind + required_frames, total_frames)
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sequence_length, dtype=int
        )

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data).float()
        video_data = video_data.permute(0, 3, 1, 2)

        video = self.norm(video_data).permute(1, 0, 2, 3)
        return {"video": video}

class MultiFilesBatchVideoSampler(BatchSampler):
    """A sampler wrapper for grouping videos within same folder into a same batch.
    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 train_folder: str = None,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket = {file_name: [] for file_name in os.listdir(self.train_folder)}
        
        #{file_name: [list(os.listdir(os.path.join(self.train_folder, file_name)))] for file_name in os.listdir(self.train_folder)}
        self.idx2file = []
        

    def __iter__(self):
        for idx in self.sampler:          
            file_name = self.idx2file[idx]
            self.bucket[file_name].append(idx)
            bucket = self.bucket[file_name]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


def group_data_fun(lengths, generator=None):
    # counter is decrease order
    counter = Counter(lengths)  # counter {'1x256x256': 3, ''}   lengths ['1x256x256', '1x256x256', '1x256x256', ...]
    grouped_indices = defaultdict(list)
    for idx, item in enumerate(lengths):  # group idx to a list
        grouped_indices[item].append(idx)

    grouped_indices = dict(grouped_indices)  # {'1x256x256': [0, 1, 2], ...}
    sorted_indices = [grouped_indices[item] for (item, _) in sorted(counter.items(), key=lambda x: x[1], reverse=True)]
    
    # shuffle in each group
    shuffle_sorted_indices = []
    for indice in sorted_indices:
        shuffle_idx = torch.randperm(len(indice), generator=generator).tolist()
        shuffle_sorted_indices.extend([indice[idx] for idx in shuffle_idx])
    return shuffle_sorted_indices

def last_group_data_fun(shuffled_megabatches, lengths):
    # lengths ['1x256x256', '1x256x256', '1x256x256' ...]
    re_shuffled_megabatches = []
    # print('shuffled_megabatches', len(shuffled_megabatches))
    for i_megabatch, megabatch in enumerate(shuffled_megabatches):
        re_megabatch = []
        for i_batch, batch in enumerate(megabatch):
            assert len(batch) != 0
                
            len_each_batch = [lengths[i] for i in batch]  # ['1x256x256', '1x256x256']
            idx_length_dict = dict([*zip(batch, len_each_batch)])  # {0: '1x256x256', 100: '1x256x256'}
            count_dict = Counter(len_each_batch)  # {'1x256x256': 2} or {'1x256x256': 1, '1x768x256': 1}
            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])  # {'1x256x256': 1, '1x768x256': 1}
                # import ipdb;ipdb.set_trace()
                # print(batch, idx_length_dict, count_dict, sorted_by_value)
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [idx for idx, length in idx_length_dict.items() if length == pick_length]
                random_select_batch = [random.choice(candidate_batch) for i in range(len(len_each_batch) - len(candidate_batch))]
                # print(batch, idx_length_dict, count_dict, sorted_by_value, pick_length, candidate_batch, random_select_batch)
                batch = candidate_batch + random_select_batch
                # print(batch)

            for i in range(1, len(batch)-1):
                # if not lengths[batch[0]] == lengths[batch[i]]:
                #     print(batch, [lengths[i] for i in batch])
                #     import ipdb;ipdb.set_trace()
                assert lengths[batch[0]] == lengths[batch[i]]
            re_megabatch.append(batch)
        re_shuffled_megabatches.append(re_megabatch)
    
    
    # for megabatch, re_megabatch in zip(shuffled_megabatches, re_shuffled_megabatches):
    #     for batch, re_batch in zip(megabatch, re_megabatch):
    #         for i, re_i in zip(batch, re_batch):
    #             if i != re_i:
    #                 print(i, re_i)
    return re_shuffled_megabatches
                
def split_to_even_chunks(megabatch, lengths, world_size, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # batch_size=2, world_size=2
    # [1, 2, 3, 4] -> [[1, 2], [3, 4]]
    # [1, 2, 3] -> [[1, 2], [3]]
    # [1, 2] -> [[1], [2]]
    # [1] -> [[1], []]
    chunks = [megabatch[i::world_size] for i in range(world_size)]

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):  
            assert batch_size > len(chunk)
            if len(chunk) != 0:  # [[1, 2], [3]] -> [[1, 2], [3, 3]]
                chunk = chunk + [random.choice(chunk) for _ in range(batch_size - len(chunk))]
            else:
                chunk = random.choice(pad_chunks)  # [[1], []] -> [[1], [1]]
                print(chunks[idx], '->', chunk)
        pad_chunks.append(chunk)
    return pad_chunks

def get_length_grouped_indices(lengths, batch_size, world_size, gradient_accumulation_size, initial_global_step, generator=None, group_data=False, seed=42):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(seed)  # every rank will generate a fixed order but random index
    # print('lengths', lengths)
    
    if group_data:
        indices = group_data_fun(lengths, generator)
    else:
        indices = torch.randperm(len(lengths), generator=generator).tolist()

    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]

    megabatches_len = [[lengths[i] for i in megabatch] for megabatch in megabatches]

    megabatches = [split_to_even_chunks(megabatch, lengths, world_size, batch_size) for megabatch in megabatches]

    split_to_even_chunks_len = [[[lengths[i] for i in batch] for batch in megabatch] for megabatch in megabatches]

    indices_mega = torch.randperm(len(megabatches), generator=generator).tolist()
    # print(f'rank {accelerator.process_index} seed {seed}, len(megabatches) {len(megabatches)}, indices_mega, {indices_mega[:50]}')
    shuffled_megabatches = [megabatches[i] for i in indices_mega]
    shuffled_megabatches_len = [[[lengths[i] for i in batch] for batch in megabatch] for megabatch in shuffled_megabatches]
    # print(f'\nrank {accelerator.process_index} sorted shuffled_megabatches_len', shuffled_megabatches_len[0], shuffled_megabatches_len[1], shuffled_megabatches_len[-2], shuffled_megabatches_len[-1])

    # import ipdb;ipdb.set_trace()
    # print('shuffled_megabatches', len(shuffled_megabatches))
    if group_data:
        shuffled_megabatches = last_group_data_fun(shuffled_megabatches, lengths)
        group_shuffled_megabatches_len = [[[lengths[i] for i in batch] for batch in megabatch] for megabatch in shuffled_megabatches]
        # print(f'\nrank {accelerator.process_index} group_shuffled_megabatches_len', group_shuffled_megabatches_len[0], group_shuffled_megabatches_len[1], group_shuffled_megabatches_len[-2], group_shuffled_megabatches_len[-1])
    
 
    #initial_global_step = initial_global_step * gradient_accumulation_size #todo

    shuffled_megabatches = shuffled_megabatches[initial_global_step:]
    #print(f'Skip the data of {initial_global_step} step!')

    return [batch for megabatch in shuffled_megabatches for batch in megabatch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        gradient_accumulation_size: int = 1, 
        initial_global_step: int = 0, 
        lengths: Optional[List[int]] = None, 
        group_data=False, 
        generator=None,
        rank: Optional[int] = None, 
        seed: int = 0, 
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.initial_global_step = initial_global_step
        self.gradient_accumulation_size = gradient_accumulation_size
        self.lengths = lengths
        self.group_data = group_data
        self.generator = generator
    
        self.rank = rank
        self.epoch = 0
  
        self.seed = seed
        
        megabatch_size = self.batch_size * self.world_size
        self.num_samples =  ((len(lengths) + megabatch_size - 1) // megabatch_size ) * self.batch_size  #todo
        #self.num_samples = self.num_samples #- self.initial_global_step * self.batch_size  * self.gradient_accumulation_size
        # print('self.lengths, self.initial_global_step, self.batch_size, self.world_size, self.gradient_accumulation_size', 
        #       len(self.lengths), self.initial_global_step, self.batch_size, self.world_size, self.gradient_accumulation_size)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        megabatch_indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, 
                                             self.gradient_accumulation_size, self.initial_global_step, 
                                             group_data=self.group_data, generator=g)
   
        # subsample
        indices = [i for batch in megabatch_indices[self.rank::self.world_size] for i in batch]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class VideoData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def _dataset(self, train):
        datasets = []
        for dataset_path, train_list, val_list in zip(self.args.data_path, self.args.train_datalist, self.args.val_datalist):
           
            dataset = MultiSizeVideoDataset(data_folder=dataset_path, data_list=train_list if train else val_list, sequence_length=self.args.sequence_length, 
                            train=train, sample_rate=self.args.sample_rate, dynamic_sample=self.args.dynamic_sample)
            datasets.append(dataset)  
        return datasets

    def _dataloader(self, train, steps = 0, batch_size = None):
        dataset = self._dataset(train)
        if isinstance(self.args.batch_size, int):
            self.args.batch_size = [self.args.batch_size]
        self.batch_size = self.args.batch_size if batch_size is None else batch_size
        assert len(dataset) == len(self.args.batch_size)
        dataloaders = []
        for dset, d_batch_size in zip(dataset, self.batch_size):
            if dist.is_initialized():
                sampler = LengthGroupedSampler(
                    batch_size=d_batch_size,
                    world_size=dist.get_world_size(), 
                    gradient_accumulation_size=1, 
                    initial_global_step=steps if train else 0, 
                    lengths=dset.lengths, 
                    group_data=True, 
                    rank = dist.get_rank()
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
        parser.add_argument('--num_workers', type=int, default=8)
        return parser

if __name__ == "__main__":
    import os
    def lines(file_path):
        with open(file_path, 'r') as file:
            return sum(1 for line in file)
    train_folder ='./kinetics-dataset/datapath'
    lengths_dict = {file_name: lines(os.path.join(train_folder, file_name)) for file_name in os.listdir(train_folder)}
    lengths = []
    for k, v in lengths_dict.items():
        lengths += [k] * min(v, 50) #(v % 7)
    world_size = 4
    sampler = []
    batch_size = 10
    for rank in range(world_size):
        sampler.append(LengthGroupedSampler(
                    batch_size=batch_size,
                    world_size=world_size, 
                    gradient_accumulation_size=1, 
                    initial_global_step=0, 
                    lengths=lengths, 
                    group_data=True, 
                    rank = rank
                ))
    
    
    with open('./sampler.txt', 'w') as f:
        for epoch in range(5):
            rank_idx = {}
            bk = []
            print(f'epoch --------------------------------------  {epoch}  ----------------------------------------------------', file=f)
            for rank in range(world_size):
                sl = sampler[rank]
                sl.set_epoch(epoch)
                for i in iter(sl):
                    bk.append(i)
                    if len(bk) == batch_size:
                        rank_idx.setdefault(f'rank_{rank}', [])
                        rank_idx[f'rank_{rank}'].append(bk)
                        bk = []
            for num in range(5):
                print('*'*5 + f'steps {num}' + '*'*5, file=f)
                for rank, bk in rank_idx.items():
                    print(f'rank {rank}: {bk[num]}', file=f)
                    print([lengths[i] for i in bk[num]], file=f)


    exit()
