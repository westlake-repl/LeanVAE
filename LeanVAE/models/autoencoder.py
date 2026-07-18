import argparse
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..modules import DiagonalGaussianDistribution, Encoder_Arch, Decoder_Arch, ISTA
from ..utils.patcher_utils import Patcher, UnPatcher


class LatentScaler():
    """Channel-wise normalization of the LeanVAE latent so that it lives in the
    same statistical space as the Wan2.2 VAE latent (48-dim).

    The mean / std below are pre-computed from the Wan2.2 VAE latent used for
    latent-space alignment training.
    """

    def __init__(self, dtype=torch.float32):
        self.mean = torch.tensor(
            [
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838,
                0.1557, -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098,
                0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665,
                -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567,
                -0.0729, 0.0899, -0.2799, -0.1230, -0.0313, -0.1649, 0.0117,
                0.0723, -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957,
                0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
            ],
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(
            [
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
            ],
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)
        self.inv_std = 1.0 / self.std

    def enc_norm(self, z: torch.Tensor) -> torch.Tensor:
        """(z - mean) / std"""
        return (z - self.mean.to(z.device)) * self.inv_std.to(z.device)

    def dec_norm(self, z: torch.Tensor) -> torch.Tensor:
        """z * std + mean"""
        return z * self.std.to(z.device) + self.mean.to(z.device)


class LeanVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim

        self.latent_bottleneck = ISTA(points_num=args.embedding_dim, out_num=args.latent_dim, iter_num=args.ista_iter_num, layer_num=args.ista_layer_num)

        # --- Wavelet transform input (restored) ---
        # The forward/inverse discrete wavelet transform is applied on the raw
        # video before the encoder / after the decoder, exactly like the
        # original LeanVAE. This replaces the identity `inp`/`out` passthrough
        # that had been used during the earlier Wan2.2-alignment experiments.
        self.dwt = Patcher()
        self.idwt = UnPatcher()

        self.encoder = Encoder_Arch(l_dim = args.l_dim, h_dim = args.h_dim, sep_num_layer = args.sep_num_layer, fusion_num_layer = args.fusion_num_layer)
        self.decoder = Decoder_Arch(l_dim = args.l_dim, h_dim = args.h_dim, sep_num_layer = args.sep_num_layer, fusion_num_layer = args.fusion_num_layer)

        self.std_layer = nn.Linear(args.embedding_dim, args.latent_dim)

        self.tile_inference = False
        self.chunksize_enc = args.chunksize_enc if hasattr(args, 'chunksize_enc') and args.chunksize_enc else 5
        self.chunksize_dec = args.chunksize_dec if hasattr(args, 'chunksize_dec') and args.chunksize_dec else 5
        # LatentScaler aligns the LeanVAE latent with the Wan2.2 VAE latent space.
        self.norm = LatentScaler()
        if args.use_tile_inference:
            self.set_tile_inference(True)
        else:
            self.set_tile_inference(False)

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, 'is_first_chunk'):
                module.is_first_chunk = is_first_chunk

    def set_tile_inference(self, tile_inference=False):
        for module in self.modules():
            if hasattr(module, 'tile_inference'):
                module.tile_inference = tile_inference

    def _build_chunk_index(self, T = 17, mtype = 'enc'):
        start_end = []
        if mtype == 'enc':
            chunksize = self.chunksize_enc
        else:
            chunksize = self.chunksize_dec
        if T >= chunksize :
            start_end.append((0, chunksize))
            start_idx = chunksize
        else:
            assert T < chunksize

        for i in range(start_idx, T, chunksize-1):
            end_idx = min(i + chunksize -1, T)
            start_end.append((i, end_idx))
        return start_end

    def encode(self, x):
        ndim = x.ndim
        if ndim == 4:
            x = x.unsqueeze(2)
            self.set_tile_inference(False)

        if self.tile_inference:
            z = []
            chunk_indexs = self._build_chunk_index(T=x.shape[2], mtype='enc')
            for idx, (start, end) in enumerate(chunk_indexs):
                if idx == 0:
                    self._set_first_chunk(True)
                else:
                    self._set_first_chunk(False)

                x_dwt = self.dwt(x[:, :, start:end])
                p = self.encoder.encode(x=x_dwt)
                z.append(self.latent_bottleneck.sample(p))
            z = torch.cat(z, dim = 1)
        else:
            x_dwt = self.dwt(x)
            p = self.encoder.encode(x=x_dwt)
            z = self.latent_bottleneck.sample(p)

        z = rearrange(z, 'b t h w d -> b d t h w')
        z = self.norm.enc_norm(z)
        return z

    def decode(self, z, is_image = False):
        z = self.norm.dec_norm(z)
        z = rearrange(z, 'b d t h w -> b t h w d')
        if is_image:
            self.set_tile_inference(False)
        if self.tile_inference:
            x_recon = []
            chunk_indexs = self._build_chunk_index(T=z.shape[1], mtype='dec')
            for idx, (start, end) in enumerate(chunk_indexs):
                if idx == 0:
                    self._set_first_chunk(True)
                else:
                    self._set_first_chunk(False)
                p_rec = self.latent_bottleneck.recon(z[:, start:end])
                x_dwt_rec = self.decoder.decode(p_rec, is_image=is_image)

                x_recon.append(self.idwt(x=x_dwt_rec))
            x_recon = torch.cat(x_recon, dim = 2)
        else:
            p_rec = self.latent_bottleneck.recon(z)
            x_dwt_rec = self.decoder.decode(p_rec, is_image=is_image)

            x_recon = self.idwt(x=x_dwt_rec)

        return x_recon



    @torch.no_grad()
    def inference(self, x):
        if x.ndim == 4 :
            is_image = True
        else:
            is_image = False
            assert x.shape[2] % 4 == 1, f"Expected frame_num % 4 == 1, but got {x.shape[2] % 4}"

        z = self.encode(x)
        x_recon = self.decode(z, is_image=is_image)

        if is_image:
            x = x.squeeze(2)
        return x, x_recon

    def forward(self, x, log_image=False):
        x_dwt = self.dwt(x)
        p = self.encoder(x=x_dwt)
        z_mean = self.latent_bottleneck.sample(p)
        z_std = self.std_layer(p)

        posterior = DiagonalGaussianDistribution(parameters=(z_mean, z_std))
        z = posterior.sample()
        p_rec = self.latent_bottleneck.recon(z)

        x_dwt_rec = self.decoder(p_rec) #b c t h w

        x_recon = self.idwt(x=x_dwt_rec)

        if log_image:
            return x, x_recon

        # NOTE: the Wan2.2-alignment training engine (autoencoder_pl.py)
        # expects a 3-tuple (x, x_recon, latent) where `latent` is the
        # Wan-normalized mean latent used for the latent-space L1 alignment loss.
        return x, x_recon, self.norm.enc_norm(rearrange(z_mean, 'b t h w d -> b d t h w'))


    @classmethod
    def load_from_checkpoint(cls, ckpt_path, device="cpu", strict=False):
        """ Load model from checkpoint, initializing args and state_dict """
        checkpoint = torch.load(ckpt_path, map_location=device)

        if "args" not in checkpoint:
            raise ValueError("Checkpoint does not contain 'args'. Ensure the checkpoint is saved correctly.")

        args = argparse.Namespace(**checkpoint["args"])

        model = cls(args)
        if "state_dict" in checkpoint:
            msg = model.load_state_dict(checkpoint["state_dict"], strict=strict)
            print(f"Successfully loaded weights from {ckpt_path}, {msg}")
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # Model architecture parameters
        parser.add_argument("--embedding_dim", type=int, default=512, help="Dimension of the embedding space.")
        parser.add_argument("--latent_dim", type=int, default=48, help="Dimension of the latent channel (48 to match Wan2.2 VAE).")
        parser.add_argument("--ista_iter_num", type=int, default=2, help="Number of iterations in ISTA latent bottleneck.")
        parser.add_argument("--ista_layer_num", type=int, default=2, help="Number of layers in ISTA latent bottleneck.")

        parser.add_argument("--l_dim", type=int, default=128)
        parser.add_argument("--h_dim", type=int, default=384)
        parser.add_argument("--sep_num_layer", type=int, default=2, help="Number of separate processing layers in encoder/decoder.")
        parser.add_argument("--fusion_num_layer", type=int, default=4, help="Number of fusion layers in encoder/decoder.")

        # Tiling inference (for memory-efficient processing)
        parser.add_argument("--use_tile_inference", action="store_true", help="Enable tiling inference to process video in chunks.")
        parser.add_argument("--chunksize_enc", type=int, default=9, help="Number of frames per chunk during tiled encoding.")
        parser.add_argument("--chunksize_dec", type=int, default=5, help="Number of frames per chunk during tiled decoding.")
        return parser
