import os
import argparse
import torch
from LeanVAE import LeanVAE
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
import tqdm
import numpy as np
from LeanVAE.models.autoencoder import LeanVAE
from LeanVAE.models.autoencoder_pl import AutoEncoderEngine
import imageio.v2 as imageio
import numpy as np

def write_video_imageio(out_path, frames, fps):
    fps = float(fps)
    if hasattr(frames, 'detach'):
        frames = frames.detach().cpu().numpy()
    if frames.shape[1] in (1,3):
        frames = np.transpose(frames, (0, 2, 3, 1))
    frames = np.clip(frames, 0, 255).astype('uint8')
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
def main(args, model, video_path, save_path, video_name):
    use_half = args.fp16
    device = args.device
    num_frames = args.sequence_length

    # if args.tile_inference:
    #     model.set_tile_inference(True)
    #     model.chunksize_enc = args.chunksize_enc if args.chunksize_enc else 5
    #     model.chunksize_dec = args.chunksize_dec if args.chunksize_dec else 5

    decord_vr = VideoReader(video_path,ctx=cpu(0))
    fps = decord_vr.get_avg_fps()
    total_frames = len(decord_vr)
    s = 0
    e = s + num_frames
    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video =  decord_vr.get_batch(frame_id_list).asnumpy()

    video = rearrange(torch.tensor(video),'t h w c -> c t h w').unsqueeze(0)
    video = video.half() if use_half else video
    regular_size = 1 # input range is [-0.5, 0.5] if regular_size = 2, [-1, 1] if regular_size = 1
    with torch.no_grad():
        video = video /( 127.5 * regular_size) - (1.0 / regular_size)
        video = video.to(device)
        x, x_rec=  model.inference_wan(video)
        x_rec = x_rec.squeeze(0).permute(1,2,3,0)
        x_rec = (torch.clamp(x_rec,-(1.0 / regular_size),(1.0 / regular_size)) + (1.0 / regular_size)) * ( 127.5 * regular_size)
        x_rec = x_rec.to('cpu', dtype=torch.uint8)
        write_video_imageio(os.path.join(save_path,  video_name), x_rec,fps=fps)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_video', type=str, default='./input_videos')
    parser.add_argument('--reconstruct_video', type=str, default='./reconstruct_videos')
    parser.add_argument('--sequence_length', type=int, default=17)
    parser.add_argument('--fp16', action='store_true')

    parser = LeanVAE.add_model_specific_args(parser)
    parser = AutoEncoderEngine.add_model_specific_args(parser)
    args = parser.parse_args()
    
    vae  = AutoEncoderEngine(args, None)
    #vae.device = args.device
    vae.setup()
   
    os.makedirs(args.reconstruct_video ,exist_ok=True)
    vae = vae.half().to(args.device) if args.fp16 else vae.to(args.device)
    for vid_name in tqdm.tqdm(os.listdir(args.input_video)):
        video_path = os.path.join(args.input_video, vid_name)
        main(args, vae, video_path, args.reconstruct_video, vid_name)
         

