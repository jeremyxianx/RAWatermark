import torch
import torch.fft as fft
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import Dataset, DataLoader
import os
import glob
from random import randint
import cv2


## watermarking method

def wm_method(img, watermark, spa_watermark, watermark_mask, freq_intensity, spa_intensity):
        
        assert img.device == watermark.device == watermark_mask.device == spa_watermark.device, "Devices must match"

        watermarkDFT = fft.fftshift(fft.fft2(watermark, dim=(-1, -2)), dim=(-1, -2))
        watermarkDFT[:] = watermarkDFT[0]

        shiftedDFT = fft.fftshift(fft.fft2(img, dim=(-1, -2)), dim=(-1, -2))

        # modifying the mask region
        shiftedDFT[:, :, watermark_mask] += freq_intensity * watermarkDFT[:, :, watermark_mask]

        # compute the inverse Fourier transform
        wm_img = fft.ifft2(fft.ifftshift(shiftedDFT, dim=(-1, -2)), dim=(-1, -2)).real 

        # also adding to the spatial domain
        wm_img += spa_intensity * spa_watermark

        return wm_img
    
    
class VideoDataset(Dataset):
    
    """
    Given a folder of *.avi(.mp4) video files organized as shown below, this dataset
    crops the video to `crop_size` and returns a continuous sequence of `no_of_frames` frames of shape.

        assets/
            video_examples/
                1. ex1.mp4
                2. ex2.mp4

    The output has shape (no_of_frames, 3, crop_size[0], crop_size[1]).
    
    Modified from: https://github.com/DAI-Lab/RivaGAN
    """

    def __init__(self, root_dir, crop_size, no_of_frames, max_crop_size=(512, 512)):
        self.no_of_frames = no_of_frames
        self.crop_size = False
        self.max_crop_size = max_crop_size
        self.video_path = []
        self.video_count = 0
        
        for ext in ["avi", "mp4"]:
            for path in glob.glob(os.path.join(root_dir, "*.%s" % ext), recursive=True):
                cap = cv2.VideoCapture(path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width == 512 and height == 512:
                    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.video_path.append((path, nb_frames))
                    self.video_count += 1
                else:
                    print("Video dimensions do not match, skipping:", path)

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        
        # Select time index
        path, nb_frames = self.video_path[idx]
        start_idx = 0
        #start_idx = randint(0, nb_frames - self.no_of_frames - 1)
        
        # Select space index
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        H, W, D = frame.shape
        x, dx, y, dy = 0, W, 0, H
        if self.crop_size:
            dy, dx = self.crop_size
            x = randint(0, W - dx - 1)
            y = randint(0, H - dy - 1)
        if self.max_crop_size[0] < dy:
            dy, dx = self.max_crop_size
            y = randint(0, H - dy - 1)
        if self.max_crop_size[1] < dx:
            dy, dx = self.max_crop_size
            x = randint(0, W - dx - 1)


        frames = []
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Append the first frame back
        frames.append(frame / 255.0)

        
        for _ in range(self.no_of_frames-1):
            ok, frame = cap.read()
            if not ok:  
                    break  
            frame = frame[y : y + dy, x : x + dx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame / 255.0)
            
        
        frames_array = np.array(frames)
        x = torch.from_numpy(frames_array).float()
        x = x.permute(0,3,1,2)
        
        return x