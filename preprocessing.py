import torch
import librosa
import os
import cv2
import time
import numpy as np

save_dir = 'RAVDESS_Model/'
video_dir = 'RAVDESS_ALL_Test'
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# max_Frame_len = 190
# max_Audio_len = 140650
# min_Frame_len = 87
# min_Audio_len = 64916

lst = []
count = 0

for subdir, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):
            count += 1
            if count % 100 == 0 :
                print(count)
            X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T,axis=0)
            
            capture = cv2.VideoCapture(os.path.join(subdir,file))

            file = int(file[7:8]) - 1

            frame_num = 16
            total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
            df = (int)(total_frame / frame_num)

            video = []
            while(capture.isOpened()):
                if capture.get(cv2.CAP_PROP_POS_FRAMES) + 1 == capture.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
                elif (capture.get(cv2.CAP_PROP_POS_FRAMES) + 1) % df != 0 :
                    ret, frame = capture.read()
                    continue
                ret, frame = capture.read()
                ''' face detection '''
                face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) != 0:
                    x,y, w, h = faces[0]
                    frame = frame[y:y+h, x:x+w]
                frame = cv2.resize(frame, (128, 128))
                video.append(frame)
                if len(video) == frame_num:
                    break
            
            arr = video, X[:64916], file, mfccs
            lst.append(arr)

video, audio, y, mfccs = zip(*lst)

video = np.asarray(video)
audio = np.asarray(audio)
y = np.asarray(y)
mfccs = np.asarray(mfccs)

np.save(save_dir + "test_all_video.npy", video)
np.save(save_dir + "test_all_audio.npy", audio)
np.save(save_dir + "test_all_y.npy", y)
np.save(save_dir + "test_all_mfccs.npy", mfccs)

print(video.shape, audio.shape, y.shape, mfccs.shape)

