import torch.nn as nn
import torch
import numpy

class fusion_net(nn.Module):
    def __init__(self, dataset, VGG_arc=False):
        super(fusion_net, self).__init__()
        self.dataset = dataset
        self.VGG_arc = VGG_arc
        if self.dataset.endswith("mfccs"):
            # 128 # 64
            self.a_conv1 = nn.Conv1d(1, 64, 3, 2, 1)
        elif self.dataset.endswith("audio"):
            # 64916
            self.a_conv1 = nn.Conv1d(1, 64, 25, 8, 1)

        self.drop_out = nn.Dropout(0.5)
        self.a_leaky_relu = nn.LeakyReLU()

        if self.dataset.endswith("mfccs"):
            # 64 #32
            self.a_conv2 = nn.Conv1d(64, 128, 3, 2, 1)
        elif self.dataset.endswith("audio"):
            # 8112
            self.a_conv2 = nn.Conv1d(64, 128, 25, 8, 1)
        self.a_bn2 = nn.BatchNorm1d(128)
        
        if self.dataset.endswith("mfccs"):
            # 32 #16
            self.a_conv3 = nn.Conv1d(128, 256, 3, 2, 1)
        elif self.dataset.endswith("audio"):
            # 1012
            self.a_conv3 = nn.Conv1d(128, 256, 25, 8, 1)
        self.a_bn3 = nn.BatchNorm1d(256)

        if self.dataset.endswith("mfccs"):
            # 16 #8
            self.a_conv4 = nn.Conv1d(256, 512, 3, 2, 1)
        elif self.dataset.endswith("audio"):
            # 124
            self.a_conv4 = nn.Conv1d(256, 512, 25, 8, 1)
        self.a_bn4 = nn.BatchNorm1d(512)

        if self.dataset.endswith("mfccs"):
            # 8 # 4
            self.a_conv5 = nn.Conv1d(512, 1024, 3, 2, 1)
        elif self.dataset.endswith("audio"):
            # 13
            self.a_conv5 = nn.Conv1d(512, 1024, 8, 2, 1)
        self.a_bn5 = nn.BatchNorm1d(1024)

        self.a_avgpool = nn.AvgPool1d(2)
        # 4096

        self.v_leaky_relu = nn.LeakyReLU()

        if not self.VGG_arc:
            # 16 * 128 * 128
            self.v_conv1 = nn.Conv3d(3, 64, [3, 3, 3], [2, 2, 1], padding=[1, 1, 1])

            self.v_conv2 = nn.Conv3d(64, 128, [3, 3, 3], [2, 2, 1], padding=[1, 1, 1])
            self.v_bn2 = nn.BatchNorm3d(128)
            
            self.v_conv3 = nn.Conv3d(128, 256, [3, 3, 3], [2, 2, 1], padding=[1, 1, 1])
            self.v_bn3 = nn.BatchNorm3d(256)

            self.v_conv4 = nn.Conv3d(256, 512, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
            self.v_bn4 = nn.BatchNorm3d(512)

            self.v_conv5 = nn.Conv3d(512, 1024, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
            self.v_bn5 = nn.BatchNorm3d(1024)

            self.v_conv6 = nn.Conv3d(1024, 2048, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
            self.v_bn6 = nn.BatchNorm3d(2048)
        else:
            self.v_conv1 = nn.Conv3d(3, 64, [3, 3, 3], [1, 1, 1], padding=[1, 1, 1])
            self.v_conv2 = nn.Conv3d(64, 64, [3, 3, 3], [2, 2, 1], padding=[1, 1, 1])

            self.v_conv3 = nn.Conv3d(64, 128, [3, 3, 3], [1, 1, 1], padding=[1, 1, 1])
            self.v_conv4 = nn.Conv3d(128, 128, [3, 3, 3], [2, 2, 1], padding=[1, 1, 1])
            self.v_bn2 = nn.BatchNorm3d(128)

            self.v_conv5 = nn.Conv3d(128, 256, [3, 3, 3], [1, 1, 1], padding=[1, 1, 1])
            self.v_conv6 = nn.Conv3d(256, 256, [3, 3, 3], [2, 2, 1], padding=[1, 1, 1])
            self.v_bn3 = nn.BatchNorm3d(256)

            self.v_conv7 = nn.Conv3d(256, 512, [3, 3, 3], [1, 1, 1], padding=[1, 1, 1])
            self.v_conv8 = nn.Conv3d(512, 512, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
            self.v_bn4 = nn.BatchNorm3d(512)

            self.v_conv9 = nn.Conv3d(512, 1024, [3, 3, 3], [1, 1, 1], padding=[1, 1, 1])
            self.v_conv10 = nn.Conv3d(1024, 1024, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
            self.v_bn5 = nn.BatchNorm3d(1024)

            self.v_conv11 = nn.Conv3d(1024, 2048, [3, 3, 3], [1, 1, 1], padding=[1, 1, 1])
            self.v_conv12 = nn.Conv3d(2048, 2048, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
            self.v_bn6 = nn.BatchNorm3d(2048)

        self.v_avgpool = nn.AvgPool3d(2)
        
        self.fc = nn.Linear(4096, 8)
        self.f_bn = nn.BatchNorm1d(8)

    def forward(self, audio, video):
        audio = self.a_conv1(audio)
        audio = self.a_leaky_relu(audio)
        audio = self.drop_out(audio)

        audio = self.a_conv2(audio)
        audio = self.a_bn2(audio)
        audio = self.a_leaky_relu(audio)
        audio = self.drop_out(audio)

        audio = self.a_conv3(audio)
        audio = self.a_bn3(audio)
        audio = self.a_leaky_relu(audio)
        audio = self.drop_out(audio)

        audio = self.a_conv4(audio)
        audio = self.a_bn4(audio)
        audio = self.a_leaky_relu(audio)
        audio = self.drop_out(audio)

        audio = self.a_conv5(audio)
        audio = self.a_bn5(audio)
        audio = self.a_leaky_relu(audio)
        audio = self.drop_out(audio)

        audio = self.a_avgpool(audio)

        audio = audio.view(audio.size(0), -1)

        if not self.VGG_arc:
            video = self.v_conv1(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv2(video)
            video = self.v_bn2(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv3(video)
            video = self.v_bn3(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv4(video)
            video = self.v_bn4(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv5(video)
            video = self.v_bn5(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv6(video)
            video = self.v_bn6(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)
        else:
            video = self.v_conv1(video)
            video = self.v_conv2(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv3(video)
            video = self.v_conv4(video)
            video = self.v_bn2(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv5(video)
            video = self.v_conv6(video)
            video = self.v_bn3(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv7(video)
            video = self.v_conv8(video)
            video = self.v_bn4(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv9(video)
            video = self.v_conv10(video)
            video = self.v_bn5(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

            video = self.v_conv11(video)
            video = self.v_conv12(video)
            video = self.v_bn6(video)
            video = self.v_leaky_relu(video)
            video = self.drop_out(video)

        video = self.v_avgpool(video)

        video = video.view(video.size(0), -1)

        feature = torch.cat((video, audio), 1)

        output = self.fc(feature)
        output = self.f_bn(output)
        return output





