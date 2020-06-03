import torch
import librosa
import os
import cv2
import time
import numpy as np

save_dir = 'RAVDESS_Model/'

if not os.path.exists(save_dir):
        os.makedirs(save_dir)

neutral = []
calm = []
happy = []
sad = []
angry = []
fearful = []
disgust = []
surprised = []

train_video = np.load(save_dir + "train_Speech_video.npy")
train_audio = np.load(save_dir + "train_Speech_audio.npy")
train_y = np.load(save_dir + "train_Speech_y.npy")
train_mfccs = np.load(save_dir + "train_Speech_mfccs.npy")

test_video = np.load(save_dir + "test_Speech_video.npy")
test_audio = np.load(save_dir + "test_Speech_audio.npy")
test_y = np.load(save_dir + "test_Speech_y.npy")
test_mfccs = np.load(save_dir + "test_Speech_mfccs.npy")


print(train_video.shape, train_audio.shape, train_y.shape, train_mfccs.shape)
print(test_video.shape, test_audio.shape, test_y.shape, test_mfccs.shape)

video = np.concatenate((train_video, test_video))
audio = np.concatenate((train_audio, test_audio))
y = np.concatenate((train_y, test_y))
mfccs = np.concatenate((train_mfccs, test_mfccs))

print(video.shape, audio.shape, y.shape, mfccs.shape)

for i in range(len(y)):
    arr = video[i], audio[i], y[i], mfccs[i]
    if y[i] == 0:
        neutral.append(arr)
    elif y[i] == 1:
        calm.append(arr)
    elif y[i] == 2:
        happy.append(arr)
    elif y[i] == 3:
        sad.append(arr)
    elif y[i] == 4:
        angry.append(arr)
    elif y[i] == 5:
        fearful.append(arr)
    elif y[i] == 6:
        disgust.append(arr)
    elif y[i] == 7:
        surprised.append(arr)

print(len(neutral), len(calm), len(happy), len(sad), len(angry), len(fearful), len(disgust), len(surprised))

neutral_video, neutral_audio, neutral_y, neutral_mfccs = zip(*neutral)
calm_video, calm_audio, calm_y, calm_mfccs = zip(*calm)
happy_video, happy_audio, happy_y, happy_mfccs = zip(*happy)
sad_video, sad_audio, sad_y, sad_mfccs = zip(*sad)
angry_video, angry_audio, angry_y, angry_mfccs = zip(*angry)
fearful_video, fearful_audio, fearful_y, fearful_mfccs = zip(*fearful)
disgust_video, disgust_audio, disgust_y, disgust_mfccs = zip(*disgust)
surprised_video, surprised_audio, surprised_y, surprised_mfccs = zip(*surprised)

neutral_video = np.asarray(neutral_video)
neutral_audio = np.asarray(neutral_audio)
neutral_y = np.asarray(neutral_y)
neutral_mfccs = np.asarray(neutral_mfccs)

calm_video = np.asarray(calm_video)
calm_audio = np.asarray(calm_audio)
calm_y = np.asarray(calm_y)
calm_mfccs = np.asarray(calm_mfccs)

happy_video = np.asarray(happy_video)
happy_audio = np.asarray(happy_audio)
happy_y = np.asarray(happy_y)
happy_mfccs = np.asarray(happy_mfccs)

sad_video = np.asarray(sad_video)
sad_audio = np.asarray(sad_audio)
sad_y = np.asarray(sad_y)
sad_mfccs = np.asarray(sad_mfccs)

angry_video = np.asarray(angry_video)
angry_audio = np.asarray(angry_audio)
angry_y = np.asarray(angry_y)
angry_mfccs = np.asarray(angry_mfccs)

fearful_video = np.asarray(fearful_video)
fearful_audio = np.asarray(fearful_audio)
fearful_y = np.asarray(fearful_y)
fearful_mfccs = np.asarray(fearful_mfccs)

disgust_video = np.asarray(disgust_video)
disgust_audio = np.asarray(disgust_audio)
disgust_y = np.asarray(disgust_y)
disgust_mfccs = np.asarray(disgust_mfccs)

surprised_video = np.asarray(surprised_video)
surprised_audio = np.asarray(surprised_audio)
surprised_y = np.asarray(surprised_y)
surprised_mfccs = np.asarray(surprised_mfccs)

train_index = np.random.choice(192, 176, replace=False)
test_index = np.delete(np.asarray(range(192)), train_index)

train_neutral_video = neutral_video[train_index]
train_neutral_audio = neutral_audio[train_index]
train_neutral_y = neutral_y[train_index]
train_neutral_mfccs = neutral_mfccs[train_index]

test_neutral_video = neutral_video[test_index]
test_neutral_audio = neutral_audio[test_index]
test_neutral_y = neutral_y[test_index]
test_neutral_mfccs = neutral_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_calm_video = calm_video[train_index]
train_calm_audio = calm_audio[train_index]
train_calm_y = calm_y[train_index]
train_calm_mfccs = calm_mfccs[train_index]

test_calm_video = calm_video[test_index]
test_calm_audio = calm_audio[test_index]
test_calm_y = calm_y[test_index]
test_calm_mfccs = calm_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_happy_video = happy_video[train_index]
train_happy_audio = happy_audio[train_index]
train_happy_y = happy_y[train_index]
train_happy_mfccs = happy_mfccs[train_index]

test_happy_video = happy_video[test_index]
test_happy_audio = happy_audio[test_index]
test_happy_y = happy_y[test_index]
test_happy_mfccs = happy_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_sad_video = sad_video[train_index]
train_sad_audio = sad_audio[train_index]
train_sad_y = sad_y[train_index]
train_sad_mfccs = sad_mfccs[train_index]

test_sad_video = sad_video[test_index]
test_sad_audio = sad_audio[test_index]
test_sad_y = sad_y[test_index]
test_sad_mfccs = sad_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_angry_video = angry_video[train_index]
train_angry_audio = angry_audio[train_index]
train_angry_y = angry_y[train_index]
train_angry_mfccs = angry_mfccs[train_index]

test_angry_video = angry_video[test_index]
test_angry_audio = angry_audio[test_index]
test_angry_y = angry_y[test_index]
test_angry_mfccs = angry_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_fearful_video = fearful_video[train_index]
train_fearful_audio = fearful_audio[train_index]
train_fearful_y = fearful_y[train_index]
train_fearful_mfccs = fearful_mfccs[train_index]

test_fearful_video = fearful_video[test_index]
test_fearful_audio = fearful_audio[test_index]
test_fearful_y = fearful_y[test_index]
test_fearful_mfccs = fearful_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_disgust_video = disgust_video[train_index]
train_disgust_audio = disgust_audio[train_index]
train_disgust_y = disgust_y[train_index]
train_disgust_mfccs = disgust_mfccs[train_index]

test_disgust_video = disgust_video[test_index]
test_disgust_audio = disgust_audio[test_index]
test_disgust_y = disgust_y[test_index]
test_disgust_mfccs = disgust_mfccs[test_index]

train_index = np.random.choice(384, 352, replace=False)
test_index = np.delete(np.asarray(range(384)), train_index)

train_surprised_video = surprised_video[train_index]
train_surprised_audio = surprised_audio[train_index]
train_surprised_y = surprised_y[train_index]
train_surprised_mfccs = surprised_mfccs[train_index]

test_surprised_video = surprised_video[test_index]
test_surprised_audio = surprised_audio[test_index]
test_surprised_y = surprised_y[test_index]
test_surprised_mfccs = surprised_mfccs[test_index]

train_video = np.concatenate((train_neutral_video, train_calm_video, train_happy_video, train_sad_video, train_angry_video, train_fearful_video, train_disgust_video, train_surprised_video))
train_audio = np.concatenate((train_neutral_audio, train_calm_audio, train_happy_audio, train_sad_audio, train_angry_audio, train_fearful_audio, train_disgust_audio, train_surprised_audio))
train_y = np.concatenate((train_neutral_y, train_calm_y, train_happy_y, train_sad_y, train_angry_y, train_fearful_y, train_disgust_y, train_surprised_y))
train_mfccs = np.concatenate((train_neutral_mfccs, train_calm_mfccs, train_happy_mfccs, train_sad_mfccs, train_angry_mfccs, train_fearful_mfccs, train_disgust_mfccs, train_surprised_mfccs))

test_video = np.concatenate((test_neutral_video, test_calm_video, test_happy_video, test_sad_video, test_angry_video, test_fearful_video, test_disgust_video, test_surprised_video))
test_audio = np.concatenate((test_neutral_audio, test_calm_audio, test_happy_audio, test_sad_audio, test_angry_audio, test_fearful_audio, test_disgust_audio, test_surprised_audio))
test_y = np.concatenate((test_neutral_y, test_calm_y, test_happy_y, test_sad_y, test_angry_y, test_fearful_y, test_disgust_y, test_surprised_y))
test_mfccs = np.concatenate((test_neutral_mfccs, test_calm_mfccs, test_happy_mfccs, test_sad_mfccs, test_angry_mfccs, test_fearful_mfccs, test_disgust_mfccs, test_surprised_mfccs))

print(train_video.shape, train_audio.shape, train_y.shape, train_mfccs.shape)
print(test_video.shape, test_audio.shape, test_y.shape, test_mfccs.shape)

np.save(save_dir + "train_final_video.npy", train_video)
np.save(save_dir + "train_final_audio.npy", train_audio)
np.save(save_dir + "train_final_mfccs.npy", train_mfccs)
np.save(save_dir + "train_final_y.npy", train_y)

np.save(save_dir + "test_final_video.npy", test_video)
np.save(save_dir + "test_final_audio.npy", test_audio)
np.save(save_dir + "test_final_mfccs.npy", test_mfccs)
np.save(save_dir + "test_final_y.npy", test_y)