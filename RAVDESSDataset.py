import torch
import os
import cv2
import time
import numpy as np

from torch.utils.data import Dataset

class RAVDESSDataset(Dataset):
    def __init__(self, dataset, train = True):
        save_dir = "RAVDESS_Model/"
        if train:
            if dataset == 'speech_mfccs':
                self.video = np.load(save_dir + "train_speech_video.npy")
                self.audio = np.load(save_dir + "train_speech_mfccs.npy")
                self.y = np.load(save_dir + "train_speech_y.npy")
            elif dataset == 'speech_audio':
                self.video = np.load(save_dir + "train_speech_video.npy")
                self.audio = np.load(save_dir + "train_speech_audio.npy")
                self.y = np.load(save_dir + "train_speech_y.npy")
            elif dataset == 'all_mfccs':
                self.video = np.load(save_dir + "train_all_video.npy")
                self.audio = np.load(save_dir + "train_all_mfccs.npy")
                self.y = np.load(save_dir + "train_all_y.npy")
            elif dataset == 'all_audio':
                self.video = np.load(save_dir + "train_all_video.npy")
                self.audio = np.load(save_dir + "train_all_audio.npy")
                self.y = np.load(save_dir + "train_all_y.npy")
            elif dataset == 'random_mfccs':
                self.video = np.load(save_dir + "train_random_video.npy")
                self.audio = np.load(save_dir + "train_random_mfccs.npy")
                self.y = np.load(save_dir + "train_random_y.npy")
            elif dataset == 'random_audio':
                self.video = np.load(save_dir + "train_random_video.npy")
                self.audio = np.load(save_dir + "train_random_audio.npy")
                self.y = np.load(save_dir + "train_random_y.npy")
            elif dataset == 'final_mfccs':
                self.video = np.load(save_dir + "train_final_video.npy")
                self.audio = np.load(save_dir + "train_final_mfccs.npy")
                self.y = np.load(save_dir + "train_final_y.npy")
            elif dataset == 'final_audio':
                self.video = np.load(save_dir + "train_final_video.npy")
                self.audio = np.load(save_dir + "train_final_audio.npy")
                self.y = np.load(save_dir + "train_final_y.npy")
            else:
                print("Dataset Error!!")
        else:
            if dataset == 'speech_mfccs':
                self.video = np.load(save_dir + "test_speech_video.npy")
                self.audio = np.load(save_dir + "test_speech_mfccs.npy")
                self.y = np.load(save_dir + "test_speech_y.npy")
            elif dataset == 'speech_audio':
                self.video = np.load(save_dir + "test_speech_video.npy")
                self.audio = np.load(save_dir + "test_speech_audio.npy")
                self.y = np.load(save_dir + "test_speech_y.npy")
            elif dataset == 'all_mfccs':
                self.video = np.load(save_dir + "test_all_video.npy")
                self.audio = np.load(save_dir + "test_all_mfccs.npy")
                self.y = np.load(save_dir + "test_all_y.npy")
            elif dataset == 'all_audio':
                self.video = np.load(save_dir + "test_all_video.npy")
                self.audio = np.load(save_dir + "test_all_audio.npy")
                self.y = np.load(save_dir + "test_all_y.npy")
            elif dataset == 'random_mfccs':
                self.video = np.load(save_dir + "test_random_video.npy")
                self.audio = np.load(save_dir + "test_random_mfccs.npy")
                self.y = np.load(save_dir + "test_random_y.npy")
            elif dataset == 'random_audio':
                self.video = np.load(save_dir + "test_random_video.npy")
                self.audio = np.load(save_dir + "test_random_audio.npy")
                self.y = np.load(save_dir + "test_random_y.npy")
            elif dataset == 'final_mfccs':
                self.video = np.load(save_dir + "test_final_video.npy")
                self.audio = np.load(save_dir + "test_final_mfccs.npy")
                self.y = np.load(save_dir + "test_final_y.npy")
            elif dataset == 'final_audio':
                self.video = np.load(save_dir + "test_final_video.npy")
                self.audio = np.load(save_dir + "test_final_audio.npy")
                self.y = np.load(save_dir + "test_final_y.npy")
            else:
                print("Dataset Error!!")
        print("Dataset Load Complete!")
        print("===========",self.video.shape,self.audio.shape,self.y.shape,"==============")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        video = torch.FloatTensor(self.video[idx])
        video = video.permute(3, 2, 1, 0)
        audio = torch.FloatTensor(self.audio[idx])
        audio = torch.unsqueeze(audio, 0)
        y = self.y[idx]
        return video, audio, y



