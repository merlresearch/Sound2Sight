# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=3):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(nn.Conv2d(nf * 8, dim, 4, 1, 0), nn.BatchNorm2d(dim), nn.Tanh())

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)  # h1
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]  #


class decoder(nn.Module):
    def __init__(self, dim, nc=3):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output


# Encoder model for audio spectrogram - MFCC
class encoder_audio(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder_audio, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 48 x 17
        self.c1 = nn.Sequential(
            nn.Conv2d(nc, nf, 4, stride=(2, 1), padding=0), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf) x 23 x 14
        self.c2 = nn.Sequential(
            nn.Conv2d(nf, nf, 4, stride=(2, 1), padding=(1, 0)), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*2) x 11 x 11
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 5 x 5
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 2 x 2
        self.c5 = nn.Sequential(nn.Conv2d(nf * 8, dim, 2, 1, 0), nn.BatchNorm2d(dim), nn.Tanh())

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


# Encoder model for audio spectrogram - STFT
class encoder_audio_stft(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder_audio_stft, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 45 x 17
        self.c1 = nn.Sequential(
            nn.Conv2d(nc, nf, 4, stride=(2, 1), padding=0), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf) x 21 x 14
        self.c2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 4, stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (nf*2) x 11 x 11
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 5 x 5
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 2 x 2
        self.c5 = nn.Sequential(nn.Conv2d(nf * 8, dim, 2, 1, 0), nn.BatchNorm2d(dim), nn.Tanh())

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class decoder_audio(nn.Module):
    def __init__(self, dim, nc=3):
        super(decoder_audio, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)  #  * 2
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)  #  * 2
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)  #  * 2
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),  #  * 2
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        vec = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        return output
