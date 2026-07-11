"""Load a 4x8x8 compression-ratio, chn=16 (latent_dim=16) LeanVAE checkpoint and
train it so that its latent space aligns with the Wan2.1 VAE latent space.

This is the training entry point of the `align-wan2.1` branch. The wavelet
patcher input is kept (see LeanVAE/models/autoencoder.py). The alignment losses
are implemented in LeanVAE/models/autoencoder_pl.py, which loads the Wan2.1 VAE
(LeanVAE/wan_base/modules/vae_2_1.py -> WanVAE, z_dim=16).

Example (single node, N GPUs):
    torchrun --nproc_per_node=N leanvae_train.py \
        --default_root_dir ./output_align_wan2_1 \
        --gpus N \
        --leanvae_ckpt /path/to/leanvae_4x8x8_chn16.pt \
        --wan_vae_pth /path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
        --grad_clip_val 1.0 --lr 5e-5 --lr_min 2e-5 --warmup_steps 10000 \
        --discriminator_iter_start 20000 --max_steps 1700000 \
        --data_path '' --train_datalist data_list --val_datalist data_list \
        --batch_size 2 --num_workers 20 --sample_rate 3 --sequence_length 17 \
        --latent_dim 16 --ista_iter_num 4 --ista_layer_num 2 \
        --l_dim 128 --h_dim 384 --sep_num_layer 2 --fusion_num_layer 4 \
        --dynamic_sample \
        [--resume_ckpt /path/to/checkpoint.ckpt]
"""
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from LeanVAE import LeanVAE
from LeanVAE.data import VideoData
from LeanVAE.models.autoencoder_pl import AutoEncoderEngine
from LeanVAE.utils.callbacks import VideoLogger, VideoLoggerWan


def main():
    pl.seed_everything(1111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Optional PL checkpoint path to resume training from.')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser = LeanVAE.add_model_specific_args(parser)
    parser = AutoEncoderEngine.add_model_specific_args(parser)

    args = parser.parse_args()

    data = VideoData(args)
    model = AutoEncoderEngine(args, data)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='train/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=5000, save_top_k=-1, filename='{epoch}-{step}-{recon_loss:.2f}'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    print("Log the reconstructed videos...")
    callbacks.append(VideoLogger(batch_frequency=5000, max_videos=4, clamp=True))
    callbacks.append(VideoLoggerWan(batch_frequency=10000, max_videos=4, clamp=True))

    logger = TensorBoardLogger(
        name=os.path.basename(args.default_root_dir),
        save_dir=args.default_root_dir,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='gpu',
        gpus=args.gpus if args.gpus is not None else 1,
        strategy='ddp',
        logger=logger,
        callbacks=callbacks,
        replace_sampler_ddp=False,  # keep the custom DDP sampler
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=49,
        max_steps=args.max_steps,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(model, data, ckpt_path=args.resume_ckpt)


if __name__ == '__main__':
    main()
