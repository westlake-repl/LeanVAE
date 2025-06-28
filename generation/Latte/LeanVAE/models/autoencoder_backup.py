import argparse
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from LeanVAE.modules import DiagonalGaussianDistribution, Encoder_Arch, Decoder_Arch, ISTA
from LeanVAE.utils.patcher_utils import Patcher, UnPatcher

class LeanVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        
        self.latent_bottleneck = ISTA(points_num=args.embedding_dim, out_num=args.latent_dim, iter_num=args.ista_iter_num, layer_num=args.ista_layer_num)
   
        self.dwt = Patcher()
        self.idwt = UnPatcher()
    
        self.encoder = Encoder_Arch(l_dim = args.l_dim, h_dim = args.h_dim, sep_num_layer = args.sep_num_layer, fusion_num_layer = args.fusion_num_layer)
        self.decoder = Decoder_Arch(l_dim = args.l_dim, h_dim = args.h_dim, sep_num_layer = args.sep_num_layer, fusion_num_layer = args.fusion_num_layer) 
        
        self.std_layer = nn.Linear(args.embedding_dim, args.latent_dim)
        
        self.tile_inference = False
        if args.use_tile_inference:
            self.set_tile_inference(True)
        else:
            self.set_tile_inference(False)
        
        
    def encode(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x_dwt = self.dwt(x)
        p = self.encoder.encode(x=x_dwt)
        z = self.latent_bottleneck.sample(p)
        z = rearrange(z, 'b t h w d -> b d t h w') 
        return z
    
    def decode(self, z, is_image = False):
        z = rearrange(z, 'b d t h w -> b t h w d')
        p_rec = self.latent_bottleneck.recon(z)
        x_dwt_rec = self.decoder.decode(p_rec, is_image=is_image) 
        
        x_recon = self.idwt(x=x_dwt_rec)
        
        return x_recon
        

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, 'is_first_chunk'):
                module.is_first_chunk = is_first_chunk
    
    def set_tile_inference(self, tile_inference=False):
        for module in self.modules():
            if hasattr(module, 'tile_inference'):
                module.tile_inference = tile_inference
    
    def inference(self, x, is_image = False):
        if x.ndim == 5 :
            B, C, T, H, W = x.shape    
        else:
            B, C,  H, W = x.shape
            T = 1  
            x = x.view(B, C, T, H, W)
        
        if self.tile_inference:
            chunksize = self.args.chunksize if self.args.chunksize else 5
            chunks = []
            chunks_rec = []
            if T >= chunksize :
                chunks.append(x[:, :, :chunksize , :, :])
                start_idx = chunksize 
            else:
                assert T < chunksize
            
            for i in range(start_idx, T, chunksize-1):
                end_idx = min(i + chunksize-1, T)
                chunks.append(x[:, :, i:end_idx, :, :])
            
            for idx, chunk in enumerate(chunks):
                if idx == 0:
                    self._set_first_chunk(True)
                else:
                    self._set_first_chunk(False)  
           
                z = self.encode(chunk)
                chunks_rec.append(self.decode(z, is_image=is_image))
            
            x_recon = torch.cat(chunks_rec, dim= 2)
        
        else:
            z = self.encode(x)
            x_recon = self.decode(z, is_image=is_image)
        
        if is_image:
            x = x.view(B, C, H, W)
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
        
        return x, x_recon, x_dwt, x_dwt_rec, posterior

    
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
        parser.add_argument("--latent_dim", type=int, default=4, help="Dimension of the latent channel.")
        parser.add_argument("--ista_iter_num", type=int, default=2, help="Number of iterations in ISTA latent bottleneck.")
        parser.add_argument("--ista_layer_num", type=int, default=2, help="Number of layers in ISTA latent bottleneck.")
        
        parser.add_argument("--l_dim", type=int, default=128)
        parser.add_argument("--h_dim", type=int, default=384)
        parser.add_argument("--sep_num_layer", type=int, default=2, help="Number of separate processing layers in encoder/decoder.")
        parser.add_argument("--fusion_num_layer", type=int, default=4, help="Number of fusion layers in encoder/decoder.")

        # Tiling inference (for memory-efficient processing)
        parser.add_argument("--use_tile_inference", action="store_true", help="Enable tiling inference to process video in chunks.")
        parser.add_argument("--chunksize", type=int, default=5)
        return parser
