import torch
import torch.nn as nn


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