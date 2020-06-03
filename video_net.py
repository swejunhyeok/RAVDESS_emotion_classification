import torch.nn as nn
import torch
import numpy

class video_net(nn.Module):
    def __init__(self, dataset, VGG_arc=False):
        super(video_net, self).__init__()
        self.dataset = dataset
        self.VGG_arc = VGG_arc
        self.v_leaky_relu = nn.LeakyReLU()
        self.drop_out = nn.Dropout(0.5)

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
        
        self.fc = nn.Linear(2048, 8)
        self.f_bn = nn.BatchNorm1d(8)

    def forward(self, video):
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

        output = self.fc(video)
        output = self.f_bn(output)
        return output





