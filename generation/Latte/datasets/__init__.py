from .sky_datasets import Sky
from torchvision import transforms
from .taichi_datasets import Taichi
from datasets import video_transforms
from .ucf101_datasets import UCF101
from .ffs_datasets import FaceForensics
from .ffs_image_datasets import FaceForensicsImages
from .sky_image_datasets import SkyImages
from .ucf101_image_datasets import UCF101Images
from .taichi_image_datasets import TaichiImages

#这里的std是不是要修改一下 原来的相当于x/125.5 - 1 在[-1,1]之间
#然而我VAE训练的是x/255 - 0.5 在-0.5，0.5之间
std_mychange = [1.0, 1.0, 1.0]
def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'ffs':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
        ])
        return FaceForensics(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ffs_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
        ])
        return FaceForensicsImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
        ])
        return UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101_img':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
        ])
        return UCF101Images(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
        ])
        return Taichi(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi_img':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
        ])
        return TaichiImages(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'sky':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
            ])
        return Sky(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'sky_img':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=std_mychange, inplace=True)
            ])
        return SkyImages(args, transform=transform_sky, temporal_sample=temporal_sample)
    else:
        raise NotImplementedError(args.dataset)