
import os
import joblib
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from video_dataset import VideoFrameDataset, ListToTensor

from sklearn.metrics import classification_report

from model import ANOMALY_CLIP_LSTM3

'''
    Training paradigm for the clip embedded LSTM models.  This version makes use of the UCF-Crime
    dataset after it has been converted into an image sequence dataset using the data_format.py script and
    subsequently had each frame converted to clip embedding written to text files.
'''

ap = ArgumentParser()
ap.add_argument('-n', '--name', type=str, required=True, help='Name used when saving the state_dict')
ap.add_argument('-e', '--epochs', required=True, type=int, help="Number of epochs to run")

#class ClipEncoder(nn.Module):
#
#    def __init__(self, device):
#        super(ClipEncoder, self).__init__()
#        self.model, self.preprocess = clip.load("ViT-B/32", device)
#        self.device = device
#
#
#    def __call__(self, img_list):
#        
#        processed = []
#
#        for img in img_list:
#
#             processed.append(self.preprocess(img).unsqueeze(0))
#
#        encoded = []
#
#        with torch.no_grad():
#
#            for img in processed:
#                                
#                encoded.append(self.model.encode_image(img.to(self.device)).float())
#        
#        encoded = torch.stack(encoded).squeeze(1)
#        
#        
#        return encoded


def fit(model, trainloader, optimizer, criterion, device):
    print("Training...")
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(trainloader), total = int(len(trainloader.dataset)/trainloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss/len(trainloader.dataset)
    train_accuracy = 100. * train_running_correct/len(trainloader.dataset)

    print(f"Train loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

    return train_loss, train_accuracy


def validate(model, validloader, criterion, device, decode_dict):
    print("Validating...")
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    ground_truth = []
    predictions = []


    with torch.no_grad():

        for i, data in tqdm(enumerate(validloader), total = int(len(validloader.dataset)/validloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()

            predictions.extend(preds)
            ground_truth.extend(target)

        val_loss = val_running_loss/len(validloader.dataset)
        val_accuracy = 100. * val_running_correct/len(validloader.dataset)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")
        
        ground_truth = [decode_dict[int(i)] for i in ground_truth]
        predictions = [decode_dict[int(i)] for i in predictions]
        print(classification_report(ground_truth, predictions))

        return val_loss, val_accuracy


def main():
    
    args = vars(ap.parse_args())

    name = args['name']
    epochs = args['epochs']
   
    # Directories
    root_dir = "/home/mtweed/Documents/New_College/S_3/Practical_DS"
    clip_ds_dir = "Embed_DS/clip_seq_dataset"
    clip_root = os.path.join(root_dir, clip_ds_dir)

    # label_dict = joblib.load(os.path.join(root, 'class_labs.pkl'))

    # label_decode = {i:lab for lab, i in label_dict.items()}
    label_decode = {0:'Normal', 1:'Anomaly'}

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameters
    lr = 1e-4
    epochs = epochs
    batch_size = 5
    num_segments = 10
    frames_per_segment = 200
    
    # Model/optimizer/le_scheduler
    model = ANOMALY_CLIP_LSTM3().to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=5,
                                                        min_lr=1e-6,
                                                        verbose=True)

    # Clip embedding transform
    # transforms = torch.nn.Sequential(
    #                        ClipEncoder(device))


    # Dataset/DataLoader definition   
    annotation_file =  "../Annot/anomaly_timestamp_embed_annotations.txt"

    dataset = VideoFrameDataset(
                root_path=clip_root,
                annotationfile_path=annotation_file,
                num_segments=num_segments,
                frames_per_segment=frames_per_segment,
                imagefile_template='frame_{:012d}',
                transform= ListToTensor(), #transforms, #ImglistToTensor(), If you want a dataset of tensors
                random_shift=True,
                test_mode=False,
                image=False
        )   
    
    # print(len(dataset))
    # import sys; sys.exit()
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [205, 68], 
                                                    torch.Generator().manual_seed(1))

    train_loader = DataLoader(train_dataset,
                            batch_size = batch_size,
                            shuffle = True,
                            )

    val_loader = DataLoader(valid_dataset, 
                            batch_size = batch_size,
                            shuffle = False)
    

    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
    
    best_acc = 0

    start = time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        # Train
        train_epoch_loss, train_epoch_accuracy = fit(model, train_loader, optimizer, criterion, device)

        # Validate
        val_epoch_loss, val_epoch_accuracy = validate(model, val_loader, criterion, device, label_decode)
       
        scheduler.step(val_epoch_loss)

        # Save state if best
        if val_epoch_accuracy > best_acc:

            best_acc = val_epoch_accuracy
            torch.save(model.state_dict(), f'../Output/state_dicts/{name}')
            
        # Track Accuracy and loss
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)



    end = time()

    print(f"{(end-start)/60:.3f} minutes")
    
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'../Output/plots/{name}_accuracy_plot.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='green', label='train loss')
    plt.plot(val_loss, color='blue', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../Output/plots/{name}_loss_plot.png')



if __name__ == "__main__":
   
    
    main()
