# LeanVAE — branch `align-wan2.2`

Train a **4×16×16, `latent_dim=48`** LeanVAE aligned with the **Wan2.2 VAE**
latent space. 


## What this branch does

Configures a 4×16×16 `latent_dim=48` LeanVAE and trains it from scratch while
aligning its latent space with the Wan2.2 VAE. Once aligned, the lightweight
LeanVAE can be used as a drop-in substitute for the (much heavier) Wan2.2 VAE
inside a Wan-based video diffusion pipeline.


### Alignment mechanism

- **`LatentScaler`** (in `autoencoder.py`): channel-wise `(z-mean)/std`
  normalization using the target Wan2.2 latent statistics, applied at the end of
  `encode()` and inverted at the start of `decode()`.
- **Training engine** (`autoencoder_pl.py`): in addition to the usual
  reconstruction / LPIPS / GAN losses it adds
  - `enc_wan_dec_lean = lean.decode(wan.encode(x))` supervised against `x`
    (L1 + LPIPS), and
  - a direct L1 loss between the LeanVAE latent and the Wan2.2 latent.
- **`LeanVAE/wan_base/`**: trimmed Wan2.2 VAE reference implementation
  (`vae.Wan2_2_VAE`).

## Train

```bash
torchrun --nproc_per_node=N leanvae_train.py \
    --default_root_dir ./output_align_wan2_2 --gpus N \
    --wan_vae_pth /path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
    --grad_clip_val 1.0 --lr 5e-5 --lr_min 1e-5 --warmup_steps 5000 \
    --discriminator_iter_start 100000 --max_steps 1700000 \
    --data_path '' --train_datalist data_list.csv --val_datalist data_list.csv \
    --batch_size 2 --num_workers 20 --sample_rate 3 --sequence_length 17 \
    --latent_dim 48 --ista_iter_num 5 --ista_layer_num 2 \
    --l_dim 128 --h_dim 384 --sep_num_layer 3 --fusion_num_layer 5 --dynamic_sample
```

## Data note

Training clips are read from a CSV whose `videoID` column holds the video paths
(see `LeanVAE/data.py`). Videos are normalized to `[-1, 1]` (`VideoNormWan`) to
match the Wan VAE input range. The `--wan_vae_pth` path must point to your local
weights.


