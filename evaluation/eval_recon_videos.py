import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
from glob import glob
import glob as gl
import pandas as pd
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval.cal_lpips import calculate_lpips
from eval.cal_fvd import calculate_fvd
from eval.cal_psnr import calculate_psnr
from eval.cal_ssim import calculate_ssim
from dataset.video_dataset import (
    ValidVideoDataset,
    DecordInit,
    Compose,
    ToTensorVideo
)

torch.manual_seed(2024)
class EvalDataset(ValidVideoDataset):
    def __init__(
        self,
        real_video_dir,
        generated_video_dir,
        num_frames,
        sample_rate=1,
    ) -> None:
        self.is_main_process = False
        self.v_decoder = DecordInit()
        self.real_video_files = []
        self.generated_video_files = self._make_dataset(generated_video_dir)
        for video_file in self.generated_video_files:
            filename = os.path.basename(video_file)
            if not os.path.exists(os.path.join(real_video_dir, filename)):
                raise Exception(os.path.join(real_video_dir, filename))
            self.real_video_files.append(os.path.join(real_video_dir, filename))
        self.num_frames = num_frames
        self.sample_rate = sample_rate
 
        self.transform = Compose(
            [
                ToTensorVideo(),
            ]
        )

    def _make_dataset(self, real_video_dir):
        samples = []
        samples += sum(
            [
                glob(os.path.join(real_video_dir, f"*.{ext}"), recursive=True)
                for ext in self.video_exts
            ],
            [],
        )
        return samples
    
    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        real_video_tensor = self._load_video(real_video_file, self.sample_rate)
        generated_video_tensor = self._load_video(generated_video_file, 1)
        return {"real": self.transform(real_video_tensor), "generated": self.transform(generated_video_tensor)}
    
    def _load_video(self, video_path, sample_rate=None):
        num_frames = self.num_frames
        if not sample_rate:
            sample_rate = self.sample_rate
        try:
            decord_vr = self.v_decoder(video_path)
        except:
            raise Exception(f"fail to load {video_path}.")
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            raise Exception("video too short!")
            
        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)
        return video_data


def calculate_common_metric(args, dataloader, device):
    score_list = []
    for batch_data in tqdm(dataloader): # {'real': real_video_tensor, 'generated':generated_video_tensor }
        real_videos = batch_data['real'] 
        generated_videos = batch_data['generated']
        assert real_videos.shape[2] == generated_videos.shape[2]
        if args.metric == 'fvd':
            tmp_list = list(calculate_fvd(real_videos, generated_videos, args.device, method=args.fvd_method)['value'].values())
        elif args.metric == 'ssim':
            tmp_list = list(calculate_ssim(real_videos, generated_videos)['value'].values())
        elif args.metric == 'psnr':
            tmp_list = list(calculate_psnr(real_videos, generated_videos)['value'].values())
        else:
            tmp_list  = list(calculate_lpips(real_videos, generated_videos, args.device)['value'].values())
        score_list += tmp_list
    return np.mean(score_list)


def main():

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    dataset = EvalDataset(
        args.real_video_dir,
        args.generated_video_dir,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
    )

    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=num_workers, pin_memory=False
    )

    metric_score = calculate_common_metric(args, dataloader, device)
    print('metric: ', args.metric, " ",metric_score)
    return metric_score


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--real_video_dir", type=str, help=("the path of original videos`"))
    parser.add_argument(
        "--generated_video_dir", type=str, help=("the path of reconstructed videos`")
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Like cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of processes to use for data loading. "
            "Defaults to `min(8, num_cpus)`"
        ),
    )
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument(
        "--metric",
        type=str,
        default="fvd",
        choices=["fvd", "psnr", "ssim", "lpips"],
    )
    parser.add_argument("--fvd_method", type=str, default='styleganv',choices=['styleganv','videogpt'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main()

