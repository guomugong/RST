import torch
import torch.nn as nn
import torch.nn.functional as F

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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)

class StyleTransform(nn.Module):
    def __init__(self, eps=1e-6, alpha=2.0, beta=0.5):
        super().__init__()
        self.eps = eps
        self.beta = torch.distributions.Uniform(0, alpha)
        self.beta2 = torch.distributions.Uniform(-beta, beta)

    def forward(self, x):
        B = x.size(0)
        C = x.size(1)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda1 = self.beta.sample((B, C, 1, 1))
        lmda1 = lmda1.to(x.device)

        lmda2 = self.beta2.sample((B, C, 1, 1))
        lmda2 = lmda2.to(x.device)

        mu_mix =  mu*lmda1 + lmda2
        sig_mix = sig*lmda1 + lmda2

        return x_normed*sig_mix + mu_mix

class RecNet(nn.Module):
    def __init__(self):
        super(RecNet, self).__init__()
        self.conv1_1 = ConvReLU(3, 256)
        self.conv2_1 = ConvReLU(256, 256)
        self.conv2_2 = ConvReLU(256, 256)
        self.conv2_3 = ConvReLU(256, 256)
        self.conv2_4 = ConvReLU(256, 256)
        self.outc= OutConv(256, 3)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        logits = self.outc(x)
        return  logits

class RecNetWithST(nn.Module):
    def __init__(self):
        super(RecNetWithST, self).__init__()
        self.conv1_1 = ConvReLU(3, 256)
        self.conv2_1 = ConvReLU(256, 256)
        self.conv2_2 = ConvReLU(256, 256)
        self.conv2_3 = ConvReLU(256, 256)
        self.conv2_4 = ConvReLU(256, 256)
        self.outc= OutConv(256, 3)
        self.st1 = StyleTransform()
        self.st2 = StyleTransform()
        self.st3 = StyleTransform()
        self.st4 = StyleTransform()
        self.st5 = StyleTransform()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.st1(x)
        x = self.conv2_1(x)
        x = self.st2(x)
        x = self.conv2_2(x)
        x = self.st3(x)
        x = self.conv2_3(x)
        x = self.st4(x)
        x = self.conv2_4(x)
        x = self.st5(x)
        logits = self.outc(x)
        return  logits
