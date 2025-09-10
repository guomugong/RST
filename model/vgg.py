import torch
import torch.nn as nn

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)

class VGG16(nn.Module):
    def __init__(self, in_channel):
        super(VGG16, self).__init__()
        self.conv1_1 = ConvReLU(in_channel, 64)
        self.conv1_2 = ConvReLU(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvReLU(64, 128)
        self.conv2_2 = ConvReLU(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvReLU(128, 256)
        self.conv3_2 = ConvReLU(256, 256)
        self.conv3_3 = ConvReLU(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvReLU(256, 512)
        self.conv4_2 = ConvReLU(512, 512)
        self.conv4_3 = ConvReLU(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvReLU(512, 512)
        self.conv5_2 = ConvReLU(512, 512)
        self.conv5_3 = ConvReLU(512, 512)

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

def vgg16(in_ch):
    model = VGG16(in_ch)
    return model
