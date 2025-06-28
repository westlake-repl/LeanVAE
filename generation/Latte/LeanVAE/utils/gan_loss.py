from typing import Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import functools
from ..modules.discriminator import NLayerDiscriminator

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def _sigmoid_cross_entropy_with_logits(labels, logits):
    """
    non-saturating loss
    """
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def non_saturate_gen_loss(logits_fake):
    """
    logits_fake: [B 1 H W]
    """
    B = logits_fake.shape[0]
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = torch.mean(logits_fake, dim=-1)
    gen_loss = torch.mean(_sigmoid_cross_entropy_with_logits(labels=torch.ones_like(logits_fake), logits=logits_fake))
    return gen_loss


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


class LeCAM_EMA(object):
    # https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/losses/vqperceptual.py
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class AdversarialLoss(nn.Module):
    def __init__(
        self,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_weight: float = 0.2,
        lecam_loss_weight: float = 0.005,
        disc_loss: str = "hinge",
        dims: int = 3,
        gen_loss_cross_entropy: bool = True,
    ):
        super().__init__()
        self.dims = dims
        assert disc_loss in ["hinge", "vanilla"]
        self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False
            ).apply(weights_init)
        
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.discriminator_weight = disc_weight
   
        self.gen_loss_cross_entropy = gen_loss_cross_entropy
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight > 0:
            self.lecam_ema = LeCAM_EMA()

    def get_trainable_parameters(self) -> Any:
        return self.discriminator.parameters()

    def forward(
        self,
        inputs,
        reconstructions,
        optimizer_idx, 
    ):
        
        if optimizer_idx == 0:
            if self.dims > 2:
                inputs, reconstructions = map(
                    lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                        (inputs, reconstructions),
                    )
          
            # generator update
            logits_fake = self.discriminator(reconstructions)
            
            if not self.gen_loss_cross_entropy:
                g_loss = -torch.mean(logits_fake)
            else:
                g_loss = non_saturate_gen_loss(logits_fake)

            g_loss = self.discriminator_weight * g_loss
            return g_loss
            

        if optimizer_idx == 1:
            inputs, reconstructions = map(
                lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                (inputs, reconstructions),
            )

            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)

            if self.lecam_loss_weight > 0:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                d_loss = lecam_loss * self.lecam_loss_weight + non_saturate_d_loss
            else:
                d_loss =  non_saturate_d_loss
            return d_loss
