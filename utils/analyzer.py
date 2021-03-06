import os
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import numpy as np
import clip
import cv2
from PIL import Image

from utils.model import CLIP_LSTM, ANOMALY_CLIP_LSTM2

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# clip encoder class
class ClipEncoder(nn.Module):
    '''
        encodes a list of images into a tensor of CLIP embeddings
    '''
    def __init__(self, device):
        super(ClipEncoder, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device)
        self.device = device


    def __call__(self, img_list):
        '''
            Input:
                img_list: list        -> A list structure containing PIL Images for each frame of the video

            Output:
                encoded: Torch tensor ->  a tensor of clip embedded images in the shape [# of frames, 512]
        '''
        # CLIP requires specific pre-processing for images
        processed = []
        print('Preprocessing frames...')
        for i, img in tqdm(enumerate(img_list), total=len(img_list)):
             processed.append(self.preprocess(img).unsqueeze(0))

        encoded = []


        with torch.no_grad():
            print('Embedding frames into the CLIP feature space...')
            for i, img in tqdm(enumerate(processed), total=len(processed)):
                encoded.append(self.model.encode_image(img.to(self.device)).float())

        encoded = torch.stack(encoded).squeeze(1)
        print('Done')
        return encoded


def frames_to_embed(input_path):
    '''
        INPUTS:
            input_path   -> Path to video file to be split into frames and clip embedded.

        OUTPUT: tensor of CLIP embedded frames
    '''


    frames = []
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    frame = Image.fromarray(frame)
    frames.append(frame)

    while ret:
        ret, frame = cap.read()
        if frame is not None:
            frame = Image.fromarray(frame)
            frames.append(frame)

    encoder = ClipEncoder(_DEVICE)
    out = encoder(frames)

    return out




def classify(video_path: str, name: str, cls_type: int=1):
    '''
        Input:
            video_path: str -> full path to video file for analysis
            name: str       -> name of video file for output
            cls_type: int   -> integer indication of analysis type {1: "Anomaly", 2: "Multi-class"}

        output:
            analysis printed to Term
    '''

    # Embed all frames using CLIP
    os.system('cls' if os.name == 'nt' else 'clear')
    embedded = frames_to_embed(video_path)

    # directory containing trained model weights
    weight_dir = os.path.join(os.getcwd(),'weights')

    # Load model based on analysis type
    if cls_type == 1:
        state_dict_path = os.path.join(weight_dir, 'anomaly_model2_full_embedDS')
        model = ANOMALY_CLIP_LSTM2().to(_DEVICE)
        model.load_state_dict(torch.load(state_dict_path))
        decoder = {0:'Anomaly', 1:'Normal'}
    else:
        state_dict_path = os.path.join(weight_dir, 'all_class_all_bal_mod')
        model = CLIP_LSTM().to(_DEVICE)
        model.load_state_dict(torch.load(state_dict_path))
        label_dict = joblib.load(os.path.join(weight_dir, 'class_labs.pkl'))

        decoder = {i:lab for lab, i in label_dict.items()}

    with torch.no_grad():

        # reduce sequence size to that used in training (300 frames)
        frames = embedded.shape[0]
        first_3rd = int(frames / 3)
        second_3rd = int(2*frames/3)

        idx = np.random.choice(range(0,first_3rd), size=100, replace=True).tolist()
        idx2 = np.random.choice(range(first_3rd, second_3rd), size=100, replace=True).tolist()
        idx3 = np.random.choice(range(second_3rd, frames), size=100, replace=True).tolist()
        idx.extend(idx2)
        idx.extend(idx3)
        idx.sort()
        embedded = embedded[idx,:]

        # batch of 5 required for analysis
        embedded = torch.stack([embedded, embedded, embedded, embedded, embedded])

        # Run model on embedded data
        out = model(embedded.to(_DEVICE))


        # calculate prediction using one on 5 outputs
        conf, pred = torch.max(out[0].data, 0)
        os.system('cls' if os.name == 'nt' else 'clear')

        # Report result using terminal output
        print(f"The model's prediction for the video '{name}' is:")
        print(f"{decoder[pred.detach().cpu().item()]} with {np.round(conf.detach().cpu().item(),3)} confidence\n\n")
