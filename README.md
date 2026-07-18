<h2 align="center"> <a href="https://arxiv.org/abs/2503.14325">LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models</a></h2>

https://github.com/user-attachments/assets/a2a4814a-192b-4cc4-b1a3-d612caa1d872

We present **LeanVAE**, a lightweight Video VAE designed for ultra-efficient video compression and scalable generation in Latent Video Diffusion Models (LVDMs).

- **Lightweight & Efficient**: Only **40M parameters**, drastically reducing FLOPs, inference time, and memory usage📉  
- **Optimized for High-Resolution Videos**: Encodes and decodes a **17-frame 1080p video** in **0.9s / 3.0s** with **6GB / 15GB** of GPU memory (under 4×8×8 / 1×8×8 compression ratios) 🎯  
- **State-of-the-Art Video Reconstruction**: Competes with leading Video VAEs 🏆  
- **Robust Support for Long Videos**：Ensures temporal consistency across varying frame lengths and inplement **lossless** temporal tiling inference for flexible processing of long sequences⏱️
- **Versatile**: Supports both **images and videos**, preserving **causality in latent space** 📽️  
- **Evidenced by Diffusion Model**: Enhances visual quality in video generation ✨  

---
## 🛠️ **Installation**
Clone the repository and install dependencies:
```
git clone https://github.com/westlake-repl/LeanVAE
cd LeanVAE
pip install -r requirements.txt
```
---


## 🌿 **Update(2026/0718) Branches: Wan VAE Latent-Space Alignment**

We update two experimental branches that
train LeanVAE so that its latent space is **aligned with the latent space of a
Wan VAE**. Aligning the latents lets a LeanVAE encoder/decoder be used as a
lightweight, drop-in substitute for the (much heavier) Wan VAE inside a Wan-based
video diffusion pipeline.

| Branch | Compression / Channels | Alignment target | 
| ------ | ---------------------- | ---------------- | 
| [`align-wan2.1`](#branch-align-wan21) | 4×8×8, `latent_dim=16` | **Wan2.1 VAE** (`z_dim=16`) | 
| [`align-wan2.2`](#branch-align-wan22) | 4×16×16, `latent_dim=48` | **Wan2.2 VAE** | 

---
## 🎯 **Quick Start** 
**Train LeanVAE**
```bash
bash scripts/train.sh
```
You can use `pl_ckpt_inference.py` to evaluate checkpoints saved during training. See this [discussion](https://github.com/westlake-repl/LeanVAE/issues/5#issuecomment-3265791491).

**Run Video Reconstruction**
```bash
bash scripts/inference.sh
```

**Evaluate Reconstruction Quality**
```bash
bash scripts/eval.sh
```
---

## 📜 **Pretrained Models**
### Video VAE Model:
| Model            | PSNR ⬆️ | LPIPS ⬇️ | Params 📦 | TFLOPs ⚡ | Checkpoint 📥                        |
| ---------------- | ------ | ------- | -------- | -------- | ----------------------------------- |
| **LeanVAE-4ch**  | 26.04  | 0.0899  | 39.8M    | 0.203    | [LeanVAE-chn4.ckpt](https://huggingface.co/Yumic/LeanVAE/resolve/main/LeanVAE-dim4.ckpt?download=true) |
| **LeanVAE-16ch** | 30.15  | 0.0461  | 39.8M    | 0.203    | [LeanVAE-chn16.ckpt](https://huggingface.co/Yumic/LeanVAE/resolve/main/LeanVAE-dim16.ckpt?download=true) |


### Latte Model:
You can find the video generation code in the `generation` folder.

| Model                    | Dataset      | FVD ⬇️  | Checkpoint 📥                        |
| ---------- | ---------- | ---------- | ----------- |
| Latte + LeanVAE-chn4 | SkyTimelapse |49.59 | [sky-chn4.ckpt](https://huggingface.co/Yumic/LeanVAE/resolve/main/sky-dim4.pt?download=true) | 
| Latte + LeanVAE-chn4 | UCF101 |164.45 | [ucf-chn4.ckpt](https://huggingface.co/Yumic/LeanVAE/resolve/main/ucf-dim4.pt?download=true) |
| Latte + LeanVAE-chn16 | SkyTimelapse |95.15 | [sky-chn16.ckpt](https://huggingface.co/Yumic/LeanVAE/resolve/main/sky-dim16.pt?download=true) |
| Latte + LeanVAE-chn16 | UCF101 |175.33 | [ucf-chn16.ckpt](https://huggingface.co/Yumic/LeanVAE/resolve/main/ucf-dim16.pt?download=true) |

---
## 🔧 **Using LeanVAE in Your Project**

```python
from LeanVAE import LeanVAE

# Load pretrained model
model = LeanVAE.load_from_checkpoint("path/to/ckpt", strict=False)

# 🔄 Encode & Decode an Image
image, image_rec = model.inference(image)

# 🖼️ Encode an image → Get latent :  
latent = model.encode(image) # (B, C, H, W) → (B, d, 1, H/8, W/8), where d=4 or 16

# 🖼️ Decode latent representation → Reconstruct image 
image = model.decode(latent, is_image=True) # (B, d, 1, H/8, W/8) → (B, C, H, W)  


# 🔄 Encode & Decode a Video
video, video_rec = model.inference(video) ## Frame count must be 4n+1 (e.g., 5, 9, 13, 17...)

# 🎞️ Encode Video → Get Latent Space
latent = model.encode(video)  # (B, C, T+1, H, W) → (B, d, T/4+1, H/8, W/8), where d=4 or 16 

# 🎞️ Decode Latent → Reconstruct Video
video = model.decode(latent) # (B, d, T/4+1, H/8, W/8) → (B, C, T+1, H, W)  

# ⚡ Enable **Temporal Tiling Inference** for Long Videos
model.set_tile_inference(True)
model.chunksize_enc = 5
model.chunksize_dec = 5
```

------


## 🔬 **Further Explorations**

- **Higher Compression Rates**   See [Issue #1](https://github.com/westlake-repl/LeanVAE/issues/1) for related discussion.
- **Diffusion on Higher Latent Channels**    Refer to *VA-VAE* : [arXiv:2501.01423](https://arxiv.org/abs/2501.01423)  *REPA-E*: [arXiv:2504.10483](https://arxiv.org/abs/2504.10483)
- **Discrete Latent Space**   Refer to *CODA*: [arXiv:2503.17760](https://arxiv.org/abs/2503.17760)



## 📂 **Preparing Data for Training**

To train LeanVAE, you need to create metadata files listing the video paths, grouped by resolution. Each file contains paths to videos of the same resolution.
```
📂 data_list
 ├── 📄 96x128.txt  📜  # Contains paths to all 96x128 videos
 │   ├── /path/to/video_1.mp4
 │   ├── /path/to/video_2.mp4
 │   ├── ...
 ├── 📄 256x256.txt  📜  # Contains paths to all 256×256 videos
 │   ├── /path/to/video_3.mp4
 │   ├── /path/to/video_4.mp4
 │   ├── ...
 ├── 📄 352x288.txt  📜  # Contains paths to all 352x288 videos
 │   ├── /path/to/video_5.mp4
 │   ├── /path/to/video_6.mp4
 │   ├── ...
```
📌 Each text file lists video paths corresponding to a specific resolution. Set `args.train_datalist` to the folder containing these files.


---
## 📜 **License**

This project is released under the **MIT License**. See the `LICENSE` file for details.


## 🔥 **Why Choose LeanVAE?**  
LeanVAE is **fast, lightweight and powerful**, enabling high-quality video compression and generation with minimal computational cost.  

If you find this work useful, consider **starring ⭐ the repository** and citing our paper!  

---

## 📝 **Cite Us**  
```bibtex
@misc{cheng2025leanvaeultraefficientreconstructionvae,
      title={LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models}, 
      author={Yu Cheng and Fajie Yuan},
      year={2025},
      eprint={2503.14325},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.14325}, 
}
```
---

## 👍 **Acknowledgement**
Our work benefits from the contributions of several open-source projects, including [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [VidTok](https://github.com/microsoft/VidTok), and [Latte](https://github.com/Vchitect/Latte). We sincerely appreciate their efforts in advancing research and open-source collaboration!
