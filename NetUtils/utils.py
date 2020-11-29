import torch
import torch.nn as nn
from selenium import webdriver


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, dropout_rate):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 32, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 32),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf * 16),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, dropout_rate):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 32),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input)
