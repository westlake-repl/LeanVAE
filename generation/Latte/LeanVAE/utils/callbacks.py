import os
import numpy as np
from PIL import Image

import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only 

import random
from .utils import save_video_grid



class VideoLogger(Callback):
    def __init__(self, batch_frequency, max_videos, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def log_local(self, save_dir, split, videos,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "videos", split)
        for k in videos:
            grid = videos[k] + 0.5
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.mp4".format(
                global_step,
                current_epoch,
                batch_idx,
                k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_video_grid(grid, path)

    def log_vid(self, pl_module, batch, batch_idx, split="train"):
        # print(batch_idx, self.batch_freq, self.check_frequency(batch_idx) and hasattr(pl_module, "log_videos") and callable(pl_module.log_videos) and self.max_videos > 0)
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_videos") and
                callable(pl_module.log_videos) and
                self.max_videos > 0):
            # print(batch_idx, self.batch_freq,  self.check_frequency(batch_idx))
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                videos = pl_module.log_videos(batch, split=split, batch_idx=batch_idx)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()
                    if self.clamp:
                        videos[k] = torch.clamp(videos[k], -0.5, 0.5)

            self.log_local(pl_module.logger.save_dir, split, videos,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch[0]['video'].ndim == 4:
            return
        self.log_vid(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_vid(pl_module, batch, batch_idx, split="val")


class DatasetCallback(Callback):
    def __init__(self, initial_batch_size, new_batch_size, step_threshold):
        self.initial_batch_size = initial_batch_size
        self.new_batch_size = new_batch_size
        self.step_threshold = step_threshold

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if trainer.global_step == self.step_threshold:
            # 更新 DataLoader 的 batch_size
            trainer.train_dataloader = trainer.video_data._dataloader(train=True, batch_size=self.new_batch_size) #  self.new_batch_size
            print(f'Batch size changed to {self.new_batch_size} at step {self.step_threshold}')

        
  