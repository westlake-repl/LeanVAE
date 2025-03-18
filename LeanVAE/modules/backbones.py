import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from typing import Tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class PEG3D(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.ds_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), groups = dim)
        self.is_first_chunk = True
        self.causal_cached = None
        self.tile_inference = False
    
    def forward(self, x):
        x = rearrange(x, 'b t h w d -> b d t h w')
        if self.tile_inference:
            if self.is_first_chunk:
                x = F.pad(x, (1, 1, 1, 1, 2, 0), value=0.) 
            else:
                x = F.pad(x, (1, 1, 1, 1, 0, 0), value=0.) 
                x = torch.concatenate((self.causal_cached, x), dim=2)
            
            self.causal_cached = x[:, :, -2:].clone()
        else:
            x = F.pad(x, (1, 1, 1, 1, 2, 0), value=0.)
        x = self.ds_conv(x.contiguous())
        x = rearrange(x, 'b d t h w -> b t h w d')
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def ffd(dim, mult=4, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )


class NAF(nn.Module):
    def __init__(self,
                 num_layer, 
                 dim,
                 ): 
        super(NAF, self).__init__()
        self.num_layer = num_layer
        self.dconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4)) 
            self.dconv_layer.append(PEG3D(dim))
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = self.dconv_layer[i](x)
            x = self.ffd_layer[i](x) 
        return x


class ResNAF(nn.Module):
    def __init__(self,
                 num_layer, 
                 dim,
                 ): 
        super(ResNAF, self).__init__()
        self.num_layer = num_layer
        self.dconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4)) 
            self.dconv_layer.append(PEG3D(dim))
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = x + self.dconv_layer[i](x)
            x = x + self.ffd_layer[i](x) 
        return x
   

class Encoder_Arch(nn.Module):
    def __init__(self,
                 l_dim = 128, 
                 h_dim = 384, 
                 sep_num_layer = 2,
                 fusion_num_layer = 4,
                 patch_size = (2,4,4),
                 in_channel = 3
                 ): 
        super(Encoder_Arch, self).__init__()
        
        self.is_first_chunk = True
        self.tile_inference = False

        self.in_channel = in_channel
        
        self._build_linear_patch(in_channel=in_channel, out_channel_low=l_dim, out_channel_high=h_dim, pt=patch_size[0], ph=patch_size[1], pw=patch_size[2])

        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim) 
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)
        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim + h_dim)
       
    def _build_linear_patch(self, in_channel = 3, out_channel_low = 128, out_channel_high = 384, pt = 2, ph = 4, pw = 4):
        patch_config = {
            'video_low': (pt, ph, pw),
            'video_high': (pt, ph, pw),
            'image_low': (1, ph, pw),
            'image_high': (1, ph, pw)
        }

        for name, (t, h, w) in patch_config.items():
            if 'low' in name:
                in_dim = in_channel * t * h * w 
                out_dim = out_channel_low 
            else:
                out_dim = out_channel_high
                in_dim = in_channel * t * h * w * 7 if 'video' in name else in_channel * t * h * w * 3
            proj = nn.Sequential(
                Rearrange(f'b c (nt {t}) (nh {h}) (nw {w}) -> b nt nh nw (c {t} {h} {w})' if 'video' in name else f'b c (nh {h}) (nw {w}) -> b 1 nh nw (c {h} {w})'),
                nn.Linear(in_dim, out_dim)
            )
            self.add_module(f"{name}_proj", proj)
    
    
    def _linear_patch(self, x, proj_type):
        low_comp, high_comp = x[:, :self.in_channel], x[:, self.in_channel:]
        return getattr(self, f"{proj_type}_low_proj")(low_comp), getattr(self, f"{proj_type}_high_proj")(high_comp)
    
    def forward(self, x):
        xi, xv = x
        xi_low, xi_high = self._linear_patch(xi, 'image')
        xv_low, xv_high = self._linear_patch(xv, 'video')
       
        low_x = torch.cat([xi_low, xv_low], dim=1)
        high_x = torch.cat([xi_high, xv_high], dim=1)
        
        high_x = self.high_layer(high_x) 
        low_x = self.low_layer(low_x)
        x = torch.cat([low_x, high_x], dim=-1)
        x = self.fusion_layer(x)
        return x

    
    
    def encode(self, x):
        xi, xv = x
        if xi is not None and xv is not None:
            xi_low, xi_high = self._linear_patch(xi, 'image')
            xv_low, xv_high = self._linear_patch(xv, 'video')
        
            low_x = torch.cat([xi_low, xv_low], dim=1)
            high_x = torch.cat([xi_high, xv_high], dim=1)
        elif xi is not None:
            low_x, high_x = self._linear_patch(xi, 'image')
        elif xv is not None:
            low_x, high_x = self._linear_patch(xv, 'video')
   
        high_x = self.high_layer(high_x) 
        low_x = self.low_layer(low_x)
        x = torch.cat([low_x, high_x], dim=-1)
        x = self.fusion_layer(x)
        return x
    
    

class Encoder_Arch(nn.Module):
    def __init__(self,
                 l_dim = 128, 
                 h_dim = 384, 
                 sep_num_layer = 2,
                 fusion_num_layer = 4,
                 patch_size = (2,4,4),
                 in_channel = 3
                 ): 
        super(Encoder_Arch, self).__init__()
        
        self.is_first_chunk = True
        self.tile_inference = False

        self.in_channel = in_channel
        
        self._build_linear_patch(in_channel=in_channel, out_channel_low=l_dim, out_channel_high=h_dim, pt=patch_size[0], ph=patch_size[1], pw=patch_size[2])

        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim) 
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)
        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim + h_dim)
       
    def _build_linear_patch(self, in_channel = 3, out_channel_low = 128, out_channel_high = 384, pt = 2, ph = 4, pw = 4):
        patch_config = {
            'video_low': (pt, ph, pw),
            'video_high': (pt, ph, pw),
            'image_low': (1, ph, pw),
            'image_high': (1, ph, pw)
        }

        for name, (t, h, w) in patch_config.items():
            if 'low' in name:
                in_dim = in_channel * t * h * w 
                out_dim = out_channel_low 
            else:
                out_dim = out_channel_high
                in_dim = in_channel * t * h * w * 7 if 'video' in name else in_channel * t * h * w * 3
            proj = nn.Sequential(
                Rearrange('b c (nt pt) (nh ph) (nw pw) -> b nt nh nw (c pt ph pw)', pt=t, ph=h, pw=w),
                nn.Linear(in_dim, out_dim)
            )
            self.add_module(f"{name}_proj", proj)
    
    
    def _linear_patch(self, x, proj_type):
        low_comp, high_comp = x[:, :self.in_channel], x[:, self.in_channel:]
        return getattr(self, f"{proj_type}_low_proj")(low_comp), getattr(self, f"{proj_type}_high_proj")(high_comp)
    
    def _feature_transform(self, low_x, high_x):
        low_x = self.low_layer(low_x)
        high_x = self.high_layer(high_x) 
        x = torch.cat([low_x, high_x], dim=-1)
        x = self.fusion_layer(x)
        return x
    
    def forward(self, x):
        xi, xv = x
        xi_low, xi_high = self._linear_patch(x=xi, proj_type='image')
        xv_low, xv_high = self._linear_patch(x=xv, proj_type='video')
       
        low_x = torch.cat([xi_low, xv_low], dim=1)
        high_x = torch.cat([xi_high, xv_high], dim=1)
        
        return self._feature_transform(low_x=low_x, high_x=high_x)

    
    
    def encode(self, x):
        xi, xv = x
        if xi is not None and xv is not None:
            xi_low, xi_high = self._linear_patch(x=xi, proj_type='image')
            xv_low, xv_high = self._linear_patch(x=xv, proj_type='video')
        
            low_x = torch.cat([xi_low, xv_low], dim=1)
            high_x = torch.cat([xi_high, xv_high], dim=1)
        elif xi is not None:
            low_x, high_x = self._linear_patch(x=xi, proj_type='image')
        elif xv is not None:
            low_x, high_x = self._linear_patch(x=xv, proj_type='video')
   
        return self._feature_transform(low_x=low_x, high_x=high_x)



class Decoder_Arch(nn.Module):
    def __init__(self,
                 l_dim = 128, 
                 h_dim = 384, 
                 sep_num_layer = 2,
                 fusion_num_layer = 4,
                 patch_size = (2,4,4),
                 in_channel = 3
                 ): 
        super(Decoder_Arch, self).__init__()
        
        self.l_dim = l_dim
        self.is_first_chunk = True
        self.tile_inference = False

        self._build_linear_unpatch(in_channel=in_channel, out_channel_low=l_dim, out_channel_high=h_dim, pt=patch_size[0], ph=patch_size[1], pw=patch_size[2])

        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim) 
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)
        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim + h_dim)
       
    
    def _build_linear_unpatch(self, in_channel = 3, out_channel_low = 128, out_channel_high = 384, pt = 2, ph = 4, pw = 4):
        patch_config = {
            'video_low': (pt, ph, pw),
            'video_high': (pt, ph, pw),
            'image_low': (1, ph, pw),
            'image_high': (1, ph, pw)
        }

        for name, (t, h, w) in patch_config.items():
            if 'low' in name:
                out_dim = in_channel * t * h * w 
                in_dim = out_channel_low 
            else:
                in_dim = out_channel_high
                out_dim = in_channel * t * h * w * 7 if 'video' in name else in_channel * t * h * w * 3
            proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                Rearrange('b nt nh nw (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)', pt=t, ph=h, pw=w),
            )
            self.add_module(f"{name}_proj", proj)
    
    def _linear_unpatch(self, x, proj_type):
        low_comp, high_comp = getattr(self, f"{proj_type}_low_proj")(x[0]), getattr(self, f"{proj_type}_high_proj")(x[1])
        return torch.cat([low_comp, high_comp], dim=1)
    
    def _feature_transform(self, x):
        x = self.fusion_layer(x)
        low_x = self.low_layer(x[:,:,:,:,:self.l_dim])
        high_x = self.high_layer(x[:,:,:,:,self.l_dim:])

        return low_x, high_x

    
    def decode(self, x, is_image = False):
        low_x, high_x = self._feature_transform(x)
        
        if is_image:
            xi = self._linear_unpatch(x=(low_x, high_x), proj_type='image')
            return (xi, None)

        else:
            if self.tile_inference and not self.is_first_chunk:
                xv = self._linear_unpatch(x=(low_x, high_x), proj_type='video')
                return (None, xv)
            else:
                xi = self._linear_unpatch(x=(low_x[:, :1], high_x[:, :1]), proj_type='image')
                xv = self._linear_unpatch(x=(low_x[:, 1:], high_x[:, 1:]), proj_type='video')
                return (xi, xv)
    
    def forward(self, x):
        low_x, high_x = self._feature_transform(x)
        xi = self._linear_unpatch(x=(low_x[:, :1], high_x[:, :1]), proj_type='image')
        xv = self._linear_unpatch(x=(low_x[:, 1:], high_x[:, 1:]), proj_type='video')
        return (xi, xv) 

class ISTA(nn.Module):
    def __init__(self,
                 points_num = 512,
                 out_num = 4,
                 iter_num = 2,
                 layer_num = 2,
                 ): 
        super(ISTA, self).__init__()
        phi_init = np.random.normal(0.0, (1 / points_num) ** 0.5, size=(out_num, points_num))
        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(np.transpose(phi_init)).float(), requires_grad=True)
        self.iter_num = iter_num
        self.forward_l = nn.ModuleList() 
        self.backward_l = nn.ModuleList() 

        for _ in range(self.iter_num):
            self.forward_l.append(NAF(num_layer=layer_num, dim=points_num))
            self.backward_l.append(NAF(num_layer=layer_num, dim=points_num))
    
        self.weights = nn.ParameterList()
        self.etas = nn.ParameterList()
        self.threshold = nn.ParameterList()
        
        for _ in range(self.iter_num):
            self.threshold.append(nn.Parameter(torch.Tensor([0.01]), requires_grad=True))
            self.weights.append(nn.Parameter(torch.tensor(1.), requires_grad=True))
            
    def sample(self, x):
        b, t, h, w, d = x.shape
        y = x.view(-1, d) @ self.phi.T    
        return y.view(b, t, h, w, -1)
    
    def recon(self, y):
        b, t, h, w, c = y.shape
        y = y.reshape(-1, c)
        recon = torch.mm(y, self.Q.t()) 
        _, d = recon.shape
        for i in range(self.iter_num):
            recon_r = recon - self.weights[i] * torch.mm((torch.mm(recon, self.phi.t()) - y), self.phi)
            recon = recon_r.reshape(b, t, h, w, -1)
            recon = self.forward_l[i](recon)
            recon = torch.mul(torch.sign(recon), F.relu(torch.abs(recon) - self.threshold[i]))
            
            recon = self.backward_l[i](recon).view(-1, d)
            recon = recon_r + recon
        return recon.view(b, t, h, w, -1)
    
    
    def forward(self, x):
        y = self.sample(x) 
        recon = self.recon(y)
        return recon
    

