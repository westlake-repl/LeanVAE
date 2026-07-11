import argparse
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from timm.models.layers import trunc_normal_
from timm.scheduler.cosine_lr import CosineLRScheduler

from .autoencoder import LeanVAE
from ..modules import LPIPS
from ..utils.gan_loss import AdversarialLoss
from ..utils.patcher_utils import Patcher
from ..wan_base.modules.vae_2_1 import WanVAE


class AutoEncoderEngine(pl.LightningModule):
    """PyTorch Lightning training engine that aligns a LeanVAE latent space
    with the Wan2.1 VAE latent space.

    On top of the standard reconstruction / LPIPS / GAN objectives it adds
    two Wan-alignment terms:
        * L1 + LPIPS between ``lean.decode(wan.encode(x))`` and ``x``, and
        * L1 between the LeanVAE latent and the Wan2.1 latent.
    """

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.video_data = data

        # Load a pretrained 4x8x8, chn=16 LeanVAE if --leanvae_ckpt is provided;
        # otherwise instantiate a fresh model.
        leanvae_ckpt = getattr(args, 'leanvae_ckpt', None)
        if leanvae_ckpt:
            self.autoencoder = LeanVAE.load_from_checkpoint(leanvae_ckpt)
        else:
            self.autoencoder = LeanVAE(args=args)

        self.automatic_optimization = False
        self.kl_weight = args.kl_weight
        self.discriminator_iter_start = args.discriminator_iter_start
        self.perceptual_weight = args.perceptual_weight
        self.l1_weight = args.l1_weight
        self.grad_clip_val = args.grad_clip_val

        if not hasattr(args, "grad_clip_val_disc"):
            args.grad_clip_val_disc = 1.0
        self.grad_clip_val_disc = args.grad_clip_val_disc

        self.perceptual_model = LPIPS().eval()
        self.perceptual_model.requires_grad_(False)

        # One discriminator for the direct reconstruction, one for the
        # Wan-encoded / LeanVAE-decoded reconstruction.
        self.gan_loss = AdversarialLoss(disc_weight=args.disc_weight, disc_num_layers=7)
        self.gan_loss_2 = AdversarialLoss(disc_weight=args.disc_weight, disc_num_layers=7)
        self.dwt_trans = Patcher()

    def setup(self, stage=None):
        if not hasattr(self, "wan_vae"):
            # Wan2.1 VAE (z_dim=16) used as the alignment target.
            self.wan_vae = WanVAE(
                vae_pth=self.args.wan_vae_pth,
                z_dim=16,
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, optimizer_idx=None, x_recon=None, log_image=False, train_gen=True):
        if log_image:
            return self.autoencoder(x, log_image)

        if optimizer_idx == 1:
            x_recon, x_recon_wan_train = x_recon
            discloss = self.gan_loss(inputs=x, reconstructions=x_recon, optimizer_idx=1)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            if x_recon_wan_train is not None:
                discloss_wan = self.gan_loss_2(inputs=x, reconstructions=x_recon_wan_train, optimizer_idx=1)
                discloss = discloss + discloss_wan
                self.log("train/discloss_wan", discloss_wan, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return discloss

        elif optimizer_idx == 0:
            cal_wan = True

            assert x.ndim == 5
            B, C, T, H, W = x.shape
            x, x_recon, posterior = self.autoencoder(x)
            recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

            x_recon_wan_train = None
            if cal_wan:
                x_enc_wan = self.wan_vae.encode_my(x, device=self.device)
                enc_wan_dec_lean = self.autoencoder.decode(x_enc_wan)
                x_recon_wan_train = enc_wan_dec_lean
                progress = min(self.global_step / 5000, 1.0)
                coef = progress ** 2

            if not train_gen:
                return x_recon, x_recon_wan_train

            g_loss = 0.0
            if self.global_step >= self.discriminator_iter_start:
                g_loss = self.gan_loss(x, x_recon, optimizer_idx=0)
                self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                if x_recon_wan_train is not None:
                    g_loss_wan = self.gan_loss_2(x, x_recon_wan_train, optimizer_idx=0)
                    self.log("train/g_loss_wan", g_loss_wan, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                    g_loss = g_loss + g_loss_wan

            # Pick k+1 aligned frames (frame 0 + 4 frames whose index mod 4 == 1)
            # for the LPIPS objective.
            k = 4
            valid_start_indices = torch.tensor([i for i in range(T - k + 1) if i % 4 == 1])
            start_idx = valid_start_indices[torch.randint(0, len(valid_start_indices), (B,))]
            frame_idx = start_idx.unsqueeze(1) + torch.arange(k)
            frame_idx = torch.cat((torch.zeros((B, 1), dtype=torch.int), frame_idx), dim=1).to(self.device)

            frame_idx_selected = frame_idx.reshape(-1, 1, k + 1, 1, 1).repeat(1, C, 1, H, W)
            frames = torch.gather(x, 2, frame_idx_selected)
            frames_recon = torch.gather(x_recon, 2, frame_idx_selected)

            frames = frames.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)
            frames_recon = frames_recon.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)

            perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            if cal_wan:
                frames_enc_wan_dec_lean = torch.gather(enc_wan_dec_lean, 2, frame_idx_selected)
                frames_enc_wan_dec_lean = frames_enc_wan_dec_lean.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)
                perceptual_loss_2 = self.perceptual_model(frames, frames_enc_wan_dec_lean).mean()
                perceptual_loss_wan = perceptual_loss_2 * self.perceptual_weight

                self.log("loss/perceptual_2", perceptual_loss_2, prog_bar=True, sync_dist=True)
                self.log("loss/perceptual_total", perceptual_loss_wan, prog_bar=True, sync_dist=True)

                # L1 alignment terms: image-space (enc_wan_dec_lean vs x) and
                # latent-space (LeanVAE latent vs Wan2.1 latent).
                l1_loss_2 = F.l1_loss(enc_wan_dec_lean, x)
                l1_loss_3 = F.l1_loss(x_enc_wan, posterior) * 0.5
                l1_loss_wan = (l1_loss_2 + l1_loss_3) * self.l1_weight

                wan_loss = perceptual_loss_wan + l1_loss_wan
                self.log("train/wan_coef", coef, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                self.log("train/wan_lpips_loss", perceptual_loss_wan, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/wan_l1_loss", l1_loss_wan, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/wan_recon_loss", wan_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("loss/l1_2_enc_dec", l1_loss_2, prog_bar=False, logger=True, on_step=True)
                self.log("loss/l1_3_latent", l1_loss_3, prog_bar=False, logger=True, on_step=True)
                self.log("loss/l1_total", l1_loss_wan, prog_bar=True, logger=True, on_step=True)
                x_recon_wan = None
            else:
                wan_loss = 0.0
                x_recon_wan = None
                enc_wan_dec_lean = None

            # Wavelet-domain reconstruction losses (low / high frequency bands).
            x_dwt = self.dwt_trans(x)
            recon_loss_low = 0.
            recon_loss_high = 0.
            for tx_rec in [x_recon, x_recon_wan, enc_wan_dec_lean]:
                if tx_rec is not None:
                    x_dwt_rec = self.dwt_trans(tx_rec)
                    recon_loss_low += (F.l1_loss(x_dwt_rec[0][:, :3], x_dwt[0][:, :3]) + F.l1_loss(x_dwt_rec[1][:, :3], x_dwt[1][:, :3])) * self.l1_weight * 0.05
                    recon_loss_high += (F.l1_loss(x_dwt_rec[0][:, 3:], x_dwt[0][:, 3:]) + F.l1_loss(x_dwt_rec[1][:, 3:], x_dwt[1][:, 3:])) * self.l1_weight * 0.1

            self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss_low", recon_loss_low, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss_high", recon_loss_high, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return perceptual_loss + recon_loss + g_loss + wan_loss + recon_loss_low + recon_loss_high, x_recon, x_recon_wan_train

        return perceptual_loss, recon_loss, 0.0

    def training_step(self, batch, batch_idx):
        x = batch[0]['video']
        cur_global_step = self.global_step

        # Alternate generator / discriminator updates every other batch.
        update_gen = (batch_idx % 2 == 0)
        update_dis = True

        def round_to_multiple(v, base=16):
            return int(round(v / base) * base)

        B, C, T, H, W = x.shape
        scale = random.choice([1, 1.25, 1.5, 2, 2.5])
        new_H, new_W = round_to_multiple(H / scale, 16), round_to_multiple(W / scale, 16)
        if (new_H != H) or (new_W != W):
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=B)

        if not torch.isfinite(x).all():
            return None

        sch1, sch2 = self.lr_schedulers()
        opt1, opt2 = self.optimizers()
        x_recon, x_recon_wan_train = None, None

        if update_gen:
            self.toggle_optimizer(opt1, optimizer_idx=0)
            loss_generator, x_recon, x_recon_wan_train = self.forward(x, optimizer_idx=0)

            if loss_generator is None or not torch.isfinite(loss_generator):
                self.untoggle_optimizer(optimizer_idx=0)
                return None

            opt1.zero_grad()
            self.manual_backward(loss_generator)
            for p in self.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = torch.zeros_like(p.grad)
            if self.grad_clip_val is not None:
                self.clip_gradients(opt1, gradient_clip_val=self.grad_clip_val)
            opt1.step()
            sch1.step(cur_global_step)
            self.untoggle_optimizer(optimizer_idx=0)

        if update_dis:
            self.toggle_optimizer(opt2, optimizer_idx=1)
            if x_recon is not None:
                loss_discriminator = self.forward(x, optimizer_idx=1, x_recon=(x_recon, x_recon_wan_train))
            else:
                x_recon, x_recon_wan_train = self.forward(x, optimizer_idx=0, train_gen=False)
                loss_discriminator = self.forward(x, optimizer_idx=1, x_recon=(x_recon, x_recon_wan_train))
            if loss_discriminator is None or not torch.isfinite(loss_discriminator):
                self.untoggle_optimizer(optimizer_idx=1)
                return None

            opt2.zero_grad()
            self.manual_backward(loss_discriminator)
            for p in self.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    self.print(f"[Warning] NaN grad in discriminator, zeroing (step {cur_global_step})")
                    p.grad = torch.zeros_like(p.grad)
            if self.grad_clip_val_disc is not None:
                self.clip_gradients(opt2, gradient_clip_val=self.grad_clip_val_disc)
            opt2.step()
            sch2.step(cur_global_step)
            self.untoggle_optimizer(optimizer_idx=1)

    def validation_step(self, batch, batch_idx):
        x = batch['video']
        perceptual_loss, recon_loss, kl_loss = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log("val/kl_loss", kl_loss, prog_bar=True)

    def train_dataloader(self):
        return self.video_data._dataloader(train=True)

    def val_dataloader(self):
        return self.video_data._dataloader(False)[0]

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(self.autoencoder.parameters(),
                                  lr=self.args.lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.gan_loss.get_trainable_parameters(),
                                    lr=self.args.lr_min, betas=(0.5, 0.9))

        lr_min = self.args.lr_min
        train_iters = self.args.max_steps - self.discriminator_iter_start
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

        sch_ae = CosineLRScheduler(
            opt_ae,
            lr_min=lr_min,
            t_initial=train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_mul=1.,
            cycle_limit=1,
            t_in_epochs=True,
        )
        sch_disc = CosineLRScheduler(
            opt_disc,
            lr_min=lr_min,
            t_initial=train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=self.args.dis_warmup_steps,
            cycle_mul=1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [opt_ae, opt_disc], [
            {"scheduler": sch_ae, "interval": "step"},
            {"scheduler": sch_disc, "interval": "step"},
        ]

    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        x = batch['video'][:4]
        x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    @torch.no_grad()
    def log_videos_wan(self, batch, **kwargs):
        """Log reconstructions using the LeanVAE decoder on Wan-encoded latents."""
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        x = batch['video'][:4]
        x_enc_wan = self.wan_vae.encode_my(x, device=self.device)
        enc_wan_dec_lean = self.autoencoder.decode(x_enc_wan)
        log["inputs"] = x
        log["reconstructions"] = enc_wan_dec_lean
        return log

    def inference_wan(self, x):
        """Run the reference Wan2.1 VAE end-to-end on ``x``."""
        if x.ndim == 4:
            is_image = True
        else:
            is_image = False
            assert x.shape[2] % 4 == 1, f"Expected frame_num % 4 == 1, but got {x.shape[2] % 4}"

        z = self.wan_vae.encode_my(x, device=self.device)
        x_recon = self.wan_vae.decode_my(z, device=self.device, log=True)

        if is_image:
            x = x.squeeze(2)
        return x, x_recon

    @torch.no_grad()
    def log_videos_wan2(self, batch, **kwargs):
        """Log reconstructions using the Wan decoder on LeanVAE-encoded latents."""
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        x = batch['video'][:4]
        z = self.autoencoder.encode(x)
        x_rec = self.wan_vae.decode_my(z, device=self.device, log=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # Training configuration
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--lr_min', type=float, default=1e-5)
        parser.add_argument('--warmup_steps', type=int, default=5000)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)
        parser.add_argument('--grad_clip_val', type=float, default=1.0)
        parser.add_argument('--grad_clip_val_disc', type=float, default=1.0)

        parser.add_argument('--kl_weight', type=float, default=1e-7)
        parser.add_argument('--perceptual_weight', type=float, default=4.)
        parser.add_argument('--l1_weight', type=float, default=4.)
        parser.add_argument('--disc_weight', type=float, default=0.2)

        # Discriminator configuration
        parser.add_argument('--dis_warmup_steps', type=int, default=0)
        parser.add_argument('--discriminator_iter_start', type=int, default=0)
        parser.add_argument('--dis_lr_multiplier', type=float, default=1.)

        # Alignment configuration
        parser.add_argument('--leanvae_ckpt', type=str, default=None,
                            help='Path to the pretrained 4x8x8 chn=16 LeanVAE checkpoint to load and align.')
        parser.add_argument('--wan_vae_pth', type=str,
                            default='./wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
                            help='Path to the Wan2.1 VAE weights used as the alignment target.')

        return parser
