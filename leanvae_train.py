import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from LeanVAE import VideoData, LeanVAE, AutoEncoderEngine
from LeanVAE.utils.callbacks import VideoLogger

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    

    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser = LeanVAE.add_model_specific_args(parser)
    parser = AutoEncoderEngine.add_model_specific_args(parser)
    
    args = parser.parse_args()

    data = VideoData(args)
    model = AutoEncoderEngine(args, data)

    if args.pretrained is not None:
        load_weights = torch.load(args.pretrained, map_location='cpu')["state_dict"]
        msg = model.load_state_dict(load_weights, strict=False)
        missing_keys = msg.missing_keys
        unexpec_keys = msg.unexpected_keys
        print(f"Model loaded from {args.pretrained}.")
        print(f"Missing: {missing_keys}")
        print(f"Unexpected: {unexpec_keys}")

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{recon_loss:.2f}'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    print("Log the reconstructed videos...")
    callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(accelerator='gpu', gpus=args.gpus)
        if args.bf16:
            kwargs = dict(accelerator='gpu', gpus=args.gpus, precision="bf16")
        if args.fp16:
            kwargs = dict(accelerator='gpu', gpus=args.gpus, precision=16)
    
    wandb_logger = WandbLogger(project="LeanVAE", name=os.path.basename(args.default_root_dir), save_dir=args.default_root_dir, config=args)
    trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=49, logger=wandb_logger, callbacks=callbacks, replace_sampler_ddp=False, limit_val_batches=0, num_sanity_val_steps=0, max_steps=args.max_steps, **kwargs)
    
    trainer.fit(model, data, ckpt_path = args.ckpt_path if hasattr(args, 'ckpt_path') else '')


if __name__ == '__main__':
    main()

