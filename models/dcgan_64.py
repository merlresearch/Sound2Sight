# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, dim, nc=1):
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
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
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


class alpha_net(nn.Module):
    def __init__(self, dim):
        super(alpha_net, self).__init__()
        self.dim = dim  # The dimension of the input representation

        self.w1 = nn.Linear(self.dim, 1)  # Linear layer for x_t
        self.w2 = nn.Linear(self.dim, 1)  # Linear layer for current state
        self.w3 = nn.Linear(self.dim, 1)  # Linear layer for smoothing term

    def forward(self, h_list, h, pos=0, opt=None):  # h_att_list
        den, att_rep = 0.0, 0.0
        # Alpha_t denominator
        for i in range(opt.n_past):
            if pos == opt.n_past:
                den += torch.exp(torch.tanh(self.w1(h_list[i]) + self.w2(h)))
            elif pos > opt.n_past:
                den += torch.exp(
                    torch.tanh(
                        self.w1(h_list[i])
                        + self.w2(h)
                        + self.w3(torch.mean(torch.stack(h_list[opt.n_past :], dim=0), dim=0))
                    )
                )  # self.w3(torch.mean(torch.cat(h_list[opt.n_past:], dim=0), dim=0))
        # Loop for iterating over the past terms
        for i in range(opt.n_past):
            if pos == opt.n_past:
                att_rep += (torch.exp(torch.tanh(self.w1(h_list[i]) + self.w2(h))) / den) * h_list[i]
            else:
                att_rep += (
                    torch.exp(
                        torch.tanh(
                            self.w1(h_list[i])
                            + self.w2(h)
                            + self.w3(torch.mean(torch.stack(h_list[opt.n_past :], dim=0), dim=0))
                        )
                    )
                    / den
                ) * h_list[
                    i
                ]  # self.w3(torch.mean(torch.cat(h_list[opt.n_past:], dim=0), dim=0))
        return att_rep


class beta_net(nn.Module):
    # Define the function for beta weighting
    def __init__(self, dim):
        super(beta_net, self).__init__()
        self.dim = dim  # The dimension of the input representation

        self.w4 = nn.Linear(self.dim, 1)  # Linear layer for x_t
        self.w5 = nn.Linear(self.dim, 1)  # Linear layer for current state
        self.w6 = nn.Linear(self.dim, 1)  # Linear layer for smoothing term

    def forward(self, h_list, h, pos=0, opt=None):  # h_att_list
        den, att_rep = 0.0, 0.0
        # Alpha_t denominator
        for i in range(opt.n_past, pos):
            den += torch.exp(
                torch.tanh(
                    self.w4(h_list[i])
                    + self.w5(h)
                    + self.w6(torch.mean(torch.stack(h_list[: opt.n_past], dim=0), dim=0))
                )
            )  # self.w3(torch.mean(torch.cat(h_list[opt.n_past:], dim=0), dim=0))

        # Loop for iterating over the past terms
        for i in range(opt.n_past, pos):
            att_rep += (
                torch.exp(
                    torch.tanh(
                        self.w4(h_list[i])
                        + self.w5(h)
                        + self.w6(torch.mean(torch.stack(h_list[: opt.n_past], dim=0), dim=0))
                    )
                )
                / den
            ) * h_list[
                i
            ]  # self.w3(torch.mean(torch.cat(h_list[opt.n_past:], dim=0), dim=0))
        return att_rep


#'''
# Define the modified Frame Predictor
class frame_predictor(nn.Module):
    # Define the function for beta weighting
    def __init__(self, ip_dim, op_dim):
        super(frame_predictor, self).__init__()
        self.ip_dim = ip_dim  # The dimension of the input representation
        self.op_dim = op_dim  # The dimension of the output representation

        self.w = nn.Linear(self.ip_dim, self.op_dim)  # Linear layer for substituting the LSTM

    def forward(self, x):  # Forward-Pass Function
        return torch.tanh(self.w(x))


#'''
class model_proj(nn.Module):
    # Define the function for beta weighting
    def __init__(self, ip_dim, op_dim):
        super(model_proj, self).__init__()
        self.ip_dim = ip_dim  # The dimension of the input representation
        self.op_dim = op_dim  # The dimension of the output representation

        self.w = nn.Linear(self.ip_dim, self.op_dim)  # Linear layer for substituting the LSTM

    def forward(self, x):  # Forward-Pass Function
        return torch.tanh(self.w(x))


class alpha_net2(nn.Module):  # This is the module for computing attention in the projection space
    def __init__(self, dim):
        super(alpha_net2, self).__init__()
        self.dim = dim  # The dimension of the input representation

        self.w1 = nn.Linear(self.dim, 1)  # Linear layer for x_t
        self.w2 = nn.Linear(self.dim, 1)  # Linear layer for current state
        self.w3 = nn.Linear(self.dim, 1)  # Linear layer for smoothing term

    def forward(self, h_list, proj_list, h_proj, pos=0, opt=None):  # h_att_list
        den, att_rep = 0.0, 0.0
        # Alpha_t denominator
        for i in range(opt.n_past):
            if pos == opt.n_past:
                den += torch.exp(torch.tanh(self.w1(proj_list[i]) + self.w2(h_proj)))
            elif pos > opt.n_past:
                den += torch.exp(
                    torch.tanh(
                        self.w1(proj_list[i])
                        + self.w2(h_proj)
                        + self.w3(torch.mean(torch.cat(proj_list[opt.n_past :], dim=0), dim=0))
                    )
                )
        # Loop for iterating over the past terms
        for i in range(opt.n_past):
            if pos == opt.n_past:
                att_rep += (torch.exp(torch.tanh(self.w1(proj_list[i]) + self.w2(h_proj))) / den) * h_list[i]
            else:
                att_rep += (
                    torch.exp(
                        torch.tanh(
                            self.w1(proj_list[i])
                            + self.w2(h_proj)
                            + self.w3(torch.mean(torch.cat(proj_list[opt.n_past :], dim=0), dim=0))
                        )
                    )
                    / den
                ) * h_list[i]
        return att_rep


class beta_net2(nn.Module):  # This is the module for computing attention in the projection space
    # Define the function for beta weighting
    def __init__(self, dim):
        super(beta_net2, self).__init__()
        self.dim = dim  # The dimension of the input representation

        self.w4 = nn.Linear(self.dim, 1)  # Linear layer for x_t
        self.w5 = nn.Linear(self.dim, 1)  # Linear layer for current state
        self.w6 = nn.Linear(self.dim, 1)  # Linear layer for smoothing term

    def forward(self, h_list, proj_list, h_proj, pos=0, opt=None):  # h_att_list
        den, att_rep = 0.0, 0.0
        # Alpha_t denominator
        for i in range(opt.n_past, pos):
            den += torch.exp(
                torch.tanh(
                    self.w4(proj_list[i])
                    + self.w5(h_proj)
                    + self.w6(torch.mean(torch.cat(proj_list[: opt.n_past], dim=0), dim=0))
                )
            )

        # Loop for iterating over the past terms
        for i in range(opt.n_past, pos):
            att_rep += (
                torch.exp(
                    torch.tanh(
                        self.w4(proj_list[i])
                        + self.w5(h_proj)
                        + self.w6(torch.mean(torch.cat(proj_list[: opt.n_past], dim=0), dim=0))
                    )
                )
                / den
            ) * h_list[i]
        return att_rep


# Define the Coupling Network
class couple(nn.Module):
    # Define the function for beta weighting
    def __init__(self, ip_dim, op_dim):

        super(couple, self).__init__()
        self.ip_dim = ip_dim  # The dimension of the input representation

        self.op_dim = op_dim  # The dimension of the output representation

        self.w = nn.Linear(
            self.ip_dim, self.op_dim
        )  # Linear layer for mapping the couple input to the output dimension

    def forward(self, x1, x2):  # Forward-Pass Function
        return torch.tanh(self.w(torch.cat((x1, x2), dim=1)))


class alpha_net_spatial(nn.Module):  # Spatial Attention with Loop
    def __init__(self, dim):
        super(alpha_net_spatial, self).__init__()
        self.dim = dim  # The dimension of the input representation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.w1 = nn.Linear(self.dim, 1)  # Linear layer for x_ij
        self.w2 = nn.Linear(self.dim, 1)  # Linear layer for x_i'j'

    def forward(self, h, opt=None):  # h_att_list
        # *den, att_rep = 0.0, 0.0

        # Compute the denominator for softmax here
        # Make the number of channels the last dimension
        h = h.transpose(1, 2).transpose(2, 3)  # ; print('h: ' + str(h.size()));
        att_rep = h.repeat(1, 1, 1, 1)  # ; print('att_rep: ' + str(att_rep.size()));
        # Compute the attention here
        for s_i in range(h.size()[1]):
            for s_j in range(h.size()[2]):
                # Gamma_t denominator
                den = (
                    torch.exp(
                        torch.tanh(
                            self.w1(h[:, s_i, s_j, :])
                            .squeeze()
                            .view(-1)
                            .unsqueeze(1)
                            .unsqueeze(1)
                            .repeat(1, h.size()[1], h.size()[2])
                            + self.w2(h[:, :, :, :]).squeeze(-1)
                        )
                    )
                    .sum(-1)
                    .sum(-1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, h.size()[1], h.size()[2])
                )
                att_rep[:, s_i, s_j, :] = (
                    (
                        (
                            torch.exp(
                                torch.tanh(
                                    self.w1(h[:, s_i, s_j, :])
                                    .squeeze()
                                    .view(-1)
                                    .unsqueeze(1)
                                    .unsqueeze(1)
                                    .repeat(1, h.size()[1], h.size()[2])
                                    + self.w2(h[:, :, :, :]).squeeze(-1)
                                )
                            )
                            / den
                        )
                        .unsqueeze(3)
                        .repeat(1, 1, 1, h.size()[3])
                        * h
                    )
                    .sum(dim=1, keepdim=True)
                    .sum(dim=2, keepdim=True)
                    .squeeze()
                )

        return self.activation(att_rep.transpose(2, 3).transpose(1, 2))


class encoder_1(nn.Module):
    def __init__(self, nc=1):
        super(encoder_1, self).__init__()
        # *self.dim = dim
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

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        # *h5 = self.c5(h4)
        return h4, [h1, h2, h3]


class encoder_2(nn.Module):
    def __init__(self, dim):
        super(encoder_2, self).__init__()
        self.dim = dim
        nf = 64
        # input size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(nn.Conv2d(nf * 8, dim, 4, 1, 0), nn.BatchNorm2d(dim), nn.Tanh())

    def forward(self, input):
        h5 = self.c5(input)
        return h5.view(-1, self.dim), None


class alpha_net_spatial2(nn.Module):
    def __init__(self, dim):
        super(alpha_net_spatial2, self).__init__()
        self.dim = dim  # The dimension of the input representation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.w1 = nn.Linear(self.dim, 1)  # Linear layer for x_ij
        self.w2 = nn.Linear(self.dim, 1)  # Linear layer for x_i'j'

    def forward(self, h, opt=None):  # h_att_list
        # Compute the denominator for softmax here
        # Make the number of channels the last dimension
        h = h.transpose(1, 2).transpose(2, 3)  # ; print('w1: ' + str(self.w1.weight.size()));
        # print('att_rep: ' + str(self.w1(h[:, :, :, :]).size()))
        # Compute the attention here
        den = (
            torch.exp(
                torch.tanh(
                    self.w1(h[:, :, :, :])
                    .squeeze(-1)
                    .flatten(start_dim=1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .repeat(1, 1, h.size()[1], h.size()[2])
                    + self.w2(h[:, :, :, :]).squeeze(-1).unsqueeze(1).repeat(1, h.size()[1] * h.size()[2], 1, 1)
                )
            )
            .sum(-1, keepdim=True)
            .sum(-2, keepdim=True)
            .repeat(1, 1, h.size()[1], h.size()[2])
        )  # Gamma_t denominator
        att_rep = (
            (
                (
                    torch.exp(
                        torch.tanh(
                            self.w1(h[:, :, :, :])
                            .squeeze(-1)
                            .flatten(start_dim=1)
                            .unsqueeze(2)
                            .unsqueeze(3)
                            .repeat(1, 1, h.size()[1], h.size()[2])
                            + self.w2(h[:, :, :, :]).squeeze(-1).unsqueeze(1).repeat(1, h.size()[1] * h.size()[2], 1, 1)
                        )
                    )
                    / den
                )
                .unsqueeze(4)
                .repeat(1, 1, 1, 1, h.size()[3])
                * h.unsqueeze(1).repeat(1, h.size()[1] * h.size()[2], 1, 1, 1)
            )
            .sum(dim=3, keepdim=True)
            .sum(dim=2, keepdim=True)
            .squeeze()
        )  # Attended representation

        return self.activation(att_rep.transpose(1, 2).reshape(-1, h.size()[3], h.size()[1], h.size()[2]))


class encoder_sptmp(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder_sptmp, self).__init__()
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
        self.c5 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 2, 1, 0),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),  # nn.Tanh()
        )
        # state size. dim x 3 x 3

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5, [h1, h2, h3, h4]  # .view(-1, self.dim)


# Define the GAP encoder here
class encoder_gap_sptmp(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder_gap_sptmp, self).__init__()
        self.dim = dim

        self.gap = nn.Sequential(nn.AvgPool2d(3, stride=1), nn.Tanh())

    def forward(self, input):
        return self.gap(input).view(-1, self.dim), None
