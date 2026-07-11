# LeanVAE — branch `align-wan2.1`

Align a **4×8×8, `latent_dim=16`** LeanVAE with the **Wan2.1 VAE** (`z_dim=16`)
latent space.

## What this branch does

Loads a pretrained 4×8×8 chn=16 LeanVAE checkpoint and fine-tunes it so its
latent space matches the Wan2.1 VAE. Once aligned, the lightweight LeanVAE can be
used as a drop-in substitute for the (much heavier) Wan2.1 VAE inside a Wan-based
video diffusion pipeline. 

### Alignment mechanism

- **`LatentScaler`** (in `autoencoder.py`): channel-wise `(z-mean)/std`
  normalization using the target Wan2.1 latent statistics, applied at the end of
  `encode()` and inverted at the start of `decode()`.
- **Training engine** (`autoencoder_pl.py`): in addition to the usual
  reconstruction / LPIPS / GAN losses it adds
  - `enc_wan_dec_lean = lean.decode(wan.encode(x))` supervised against `x`
    (L1 + LPIPS), and
  - a direct L1 loss between the LeanVAE latent and the Wan2.1 latent.
- **`LeanVAE/wan_base/`**: trimmed Wan2.1 VAE reference implementation
  (`vae_2_1.WanVAE`).

## Train

```bash
torchrun --nproc_per_node=N leanvae_train.py \
    --default_root_dir ./output_align_wan2_1 --gpus N \
    --leanvae_ckpt  /path/to/leanvae_4x8x8_chn16.pt \
    --wan_vae_pth   /path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
    --grad_clip_val 1.0 --lr 5e-5 --lr_min 2e-5 --warmup_steps 10000 \
    --discriminator_iter_start 50000 --max_steps 1700000 \
    --data_path '' --train_datalist data_list.csv --val_datalist data_list.csv \
    --batch_size 2 --num_workers 20 --sample_rate 3 --sequence_length 17 \
    --latent_dim 16 --ista_iter_num 4 --ista_layer_num 2 \
    --l_dim 128 --h_dim 384 --sep_num_layer 2 --fusion_num_layer 4 --dynamic_sample
```

## Reconstruct

```bash
python pl_ckpt_inference.py --ckpt /path/to/trained.ckpt ...
```

## Data note

Training clips are read from a CSV whose `videoID` column holds the video paths
(see `LeanVAE/data.py`). Videos are normalized to `[-1, 1]` (`VideoNormWan`) to
match the Wan VAE input range. The `--wan_vae_pth` / `--leanvae_ckpt` paths must
point to your local weights.

