import numpy as np
import torch
import cv2
import os
import time
from torchvision import transforms

#from utils.dataset_tools import VideoDataset
from scripts.tools import wm_method
from scripts.model_tools import get_model
    
class RAWatermark(object):
    
    def __init__(self, device = 'cpu', model_type = 'resnet18', save_dir = 'assets/pre_trained', wm_index = 0):
        
        ## setup watermark
        self.spa_watermark = torch.load(os.path.join(save_dir, 'wm{}/wm_{}.pt'.format(wm_index, wm_index))).to(device)
        self.freq_watermark = torch.load(os.path.join(save_dir, 'wm{}/spa_wm_{}.pt'.format(wm_index, wm_index))).to(device)
        self.wm_mask = torch.load(os.path.join(save_dir, 'wm{}/wm_mask_{}.pt'.format(wm_index,wm_index))).to(device)
        self.spa_intensity = 0.05
        self.freq_intensity = 0.05
        self.wm_clip = 1.0
        
        ## setup classifer
        self.classifier = get_model(model_type).to(device)
        self.classifier.load_state_dict(torch.load(os.path.join(save_dir, 'wm{}/model_{}.pth'.format(wm_index,wm_index))))


        
    def encode(self, video, injection_every_k_frames = 1):

        """
        
        This function takes a video and injects the watermark into it. The watermark is injected into every k-th frame of the video.
        
        """
        
        # Check the size of video. Currently, we only support videos in (F, H = 512, W = 512, C = 3) format where 
        # F is the number of total frames, H (= 512) is the height, W (= 512) is the width, and C (= 3) is the number of channels.
        shape = video.size()
        if len(shape) == 4 and shape[2] == shape[3] == 512 and shape[1] == 3:
            pass
        else:
            raise ValueError('Input shape is not in the correct form. Currently, we only support videos in (number of frames, C = 3, H = 512, W = 512)')
        
    
        # Select every k-th frame
        selected_frames = video[::injection_every_k_frames]
        unselected_frames = [frame for index, frame in enumerate(video) if index % injection_every_k_frames != 0]

        # Embed watermark and clamp values to a specified range
        wm_video_selected = wm_method(selected_frames, self.freq_watermark, self.spa_watermark, self.wm_mask, self.freq_intensity, self.spa_intensity)
        wm_video_selected = wm_video_selected.clamp(0, self.wm_clip)

        # Combine selected elements and unselected elements in their original order

        combined_frames = [wm_video_selected[index // injection_every_k_frames] if index % injection_every_k_frames == 0 else unselected_frames[index // injection_every_k_frames] for index, _ in enumerate(video)]

        # convert list to tensor
        combined_frames = torch.stack(combined_frames)

        return combined_frames
    
    
    def encode_img(self, img):
        
        """
        
        This function takes an image and injects the watermark into it.
        
        """
        
        # Embed watermark and clamp values to a specified range
        wm_img = wm_method(img, self.freq_watermark, self.spa_watermark, self.wm_mask, self.freq_intensity, self.spa_intensity)
        wm_img = wm_img.clamp(0, self.wm_clip)
        
        return wm_img
    
    
    def detect(self, video, decision_thres = 0.5):
        
        """
        
        This function takes a video and detects the watermark in it.
        
        """
        
        # Check the size of video. Currently, we only support videos in (F, H = 512, W = 512, C = 3) format where 
        # F is the number of total frames, H (= 512) is the height, W (= 512) is the width, and C (= 3) is the number of channels.
        shape = video.size()
        if len(shape) == 4 and shape[2] == shape[3] == 512 and shape[1] == 3:
            pass
        else:
            raise ValueError('Input shape is not in the correct form. Currently, we only support videos in (number of frames, C = 3, H = 512, W = 512)')
        
        # switch to evaluation mode
        self.classifier.eval()
        

        with torch.no_grad():
            pred, _ = self.classifier(video)
            
        print('Watermark detected!' if torch.mean(torch.softmax(pred, dim=1)[:, 1]).item() > decision_thres else 'Watermark not detected!')


    def detect_img(self, img, decision_thres = 0.5, prob = False):
        
        """
        
        This function takes an image and detects the watermark in it.
        
        """
        
        # switch to evaluation mode
        self.classifier.eval()
        
        # convert to tensor
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred, _ = self.classifier(img)
            
        if prob:
            return torch.softmax(pred, dim=1)[:, 1].item()
        else:
            print('Watermark detected!' if (torch.softmax(pred, dim=1)[:, 1]).item() > decision_thres else 'Watermark not detected!')   
        

        

