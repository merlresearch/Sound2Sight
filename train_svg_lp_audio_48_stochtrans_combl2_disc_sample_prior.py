# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import pickle
import random

import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models.lstm as lstm_models
import models.transformers as transformer_models
import utils3_48 as utils

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.002, type=float, help="learning rate")
parser.add_argument("--disc_lr", default=0.0025, type=float, help="learning rate for the discriminator")
parser.add_argument("--beta1", default=0.9, type=float, help="momentum term for adam")
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--log_dir", default="logs/lp", help="base directory to save logs")
parser.add_argument("--model_dir", default="", help="base directory to save logs")
parser.add_argument("--name", default="", help="identifier for directory")
parser.add_argument("--data_root", default="data", help="root directory for data")
parser.add_argument("--optimizer", default="adam", help="optimizer to train with")
parser.add_argument("--niter", type=int, default=300, help="number of epochs to train for")
parser.add_argument("--seed", default=1, type=int, help="manual seed")
parser.add_argument("--epoch_size", type=int, default=600, help="epoch size")
parser.add_argument("--image_width", type=int, default=48, help="the height / width of the input image to network")
parser.add_argument("--channels", default=1, type=int)
parser.add_argument("--dataset", default="smmnist", help="dataset to train with")
parser.add_argument("--n_past", type=int, default=30, help="number of frames to condition on")
parser.add_argument("--n_future", type=int, default=30, help="number of frames to predict during training")
parser.add_argument("--n_eval", type=int, default=60, help="number of frames to predict during eval")
parser.add_argument("--rnn_size", type=int, default=256, help="dimensionality of input features of the transformers")
parser.add_argument(
    "--disc_rnn_size", type=int, default=256, help="dimensionality of hidden layer of discriminator LSTM"
)
parser.add_argument("--prior_rnn_layers", type=int, default=1, help="number of layers of the Transformer")
parser.add_argument("--posterior_rnn_layers", type=int, default=1, help="number of layers of the Transformer")
parser.add_argument("--audio_rnn_layers", type=int, default=1, help="number of layers of the audio only Transformer")
parser.add_argument(
    "--video_rnn_layers",
    type=int,
    default=1,
    help="number of layers of the video only Transformer in the z estimation block",
)
parser.add_argument("--disc_layers", type=int, default=1, help="number of layers of the Discriminator LSTM")
parser.add_argument("--predictor_rnn_layers", type=int, default=2, help="number of layers of the Transformer")
parser.add_argument("--dim_ff", type=int, default=128, help="dimension of the feedforward layer of the Transformer")
parser.add_argument("--nhead", type=int, default=4, help="number of multi-heads in the Transformer")
parser.add_argument("--z_dim", type=int, default=10, help="dimensionality of z_t")
parser.add_argument(
    "--g_dim", type=int, default=128, help="dimensionality of encoder output vector and decoder input vector"
)
parser.add_argument("--beta", type=float, default=0.0001, help="weighting on KL to prior")
parser.add_argument("--beta_gen_loss", type=float, default=0.0001, help="weighting on the generator loss")
parser.add_argument("--model", default="dcgan", help="model type (dcgan | vgg)")
parser.add_argument(
    "--disc_buffer", type=int, default=3, help="number of frames to be kept in the buffer discriminator"
)
parser.add_argument("--data_threads", type=int, default=5, help="number of data loading threads")
parser.add_argument("--num_digits", type=int, default=1, help="number of digits for moving mnist")
parser.add_argument(
    "--last_frame_skip",
    action="store_true",
    help="if true, skip connections go between frame t and frame t+t rather than last ground truth frame",
)

opt = parser.parse_args()
if opt.model_dir != "":
    # load model and continue training from checkpoint
    saved_model = torch.load("%s/model.pth" % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model["opt"]
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = "%s/continued" % opt.log_dir
    saved_opt = torch.load("%s/opt.pth" % opt.model_dir)
else:
    name = (
        "stoch_trans_combl2_disc_model=%s%dx%d-rnn_size=%d-buff_size=%d-n_ff=%d-n_head=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-disc_lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s"
        % (
            opt.model,
            opt.image_width,
            opt.image_width,
            opt.rnn_size,
            opt.disc_buffer,
            opt.dim_ff,
            opt.nhead,
            opt.predictor_rnn_layers,
            opt.posterior_rnn_layers,
            opt.prior_rnn_layers,
            opt.n_past,
            opt.n_future,
            opt.lr,
            opt.disc_lr,
            opt.g_dim,
            opt.z_dim,
            opt.last_frame_skip,
            opt.beta,
            opt.name,
        )
    )
    if opt.dataset == "smmnist":
        opt.log_dir = "%s/%s-%d/%s" % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = "%s/%s/%s" % (opt.log_dir, opt.dataset, name)

os.makedirs("%s/gen/" % opt.log_dir, exist_ok=True)
os.makedirs("%s/plots/" % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == "adam":
    opt.optimizer = optim.Adam
elif opt.optimizer == "rmsprop":
    opt.optimizer = optim.RMSprop
elif opt.optimizer == "sgd":
    opt.optimizer = optim.SGD
else:
    raise ValueError("Unknown optimizer: %s" % opt.optimizer)


if opt.model_dir != "":
    frame_predictor = saved_model["frame_predictor"]
    video_rnn_posterior = saved_model["video_rnn_posterior"]
    video_rnn_prior = saved_model["video_rnn_prior"]
    audio_rnn = saved_model["audio_rnn"]
    posterior = saved_model["posterior"]
    prior = saved_model["prior"]
    disc = saved_model["disc"]
else:
    # Prediction Network
    frame_predictor = lstm_models.lstm(
        opt.g_dim + opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size
    )
    # Stochastic Network
    # Transformer Components
    video_rnn_posterior = transformer_models.transformer(
        opt.g_dim, opt.g_dim, opt.rnn_size, opt.video_rnn_layers, opt.batch_size, opt.dim_ff, opt.nhead
    )
    video_rnn_prior = transformer_models.transformer(
        opt.g_dim, opt.g_dim, opt.rnn_size, opt.video_rnn_layers, opt.batch_size, opt.dim_ff, opt.nhead
    )
    audio_rnn = transformer_models.transformer(
        opt.g_dim, opt.g_dim, opt.rnn_size, opt.audio_rnn_layers, opt.batch_size, opt.dim_ff, opt.nhead
    )
    # LSTM Components
    posterior = lstm_models.gaussian_lstm(
        2 * opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size
    )
    prior = lstm_models.gaussian_lstm(2 * opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    disc = lstm_models.lstm(2 * opt.g_dim, 2, opt.disc_rnn_size, opt.disc_layers, opt.batch_size)
    disc_std = nn.Linear(opt.g_dim, 2)
    frame_predictor.apply(utils.init_weights)
    video_rnn_posterior.apply(utils.init_weights)
    video_rnn_prior.apply(utils.init_weights)
    audio_rnn.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)
    disc.apply(utils.init_weights)
    disc_std.apply(utils.init_weights)

if opt.model == "dcgan":
    if opt.image_width == 64:
        import models.dcgan_64 as model
    elif opt.image_width == 48:
        import models.dcgan_48 as model
    elif opt.image_width == 96:
        import models.dcgan_96 as model
    elif opt.image_width == 128:
        import models.dcgan_128 as model
elif opt.model == "vgg":
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError("Unknown model: %s" % opt.model)

if opt.model_dir != "":
    decoder = saved_model["decoder"]
    encoder = saved_model["encoder"]
    audio_enc = saved_model["audio_enc"]
    disc_encoder = saved_model["disc_encoder"]
    disc_audio_enc = saved_model["disc_audio_enc"]
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    audio_enc = model.encoder_audio_stft(opt.g_dim, opt.channels)
    disc_encoder = model.encoder(opt.g_dim, opt.channels)
    disc_audio_enc = model.encoder_audio_stft(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)
    audio_enc.apply(utils.init_weights)
    disc_encoder.apply(utils.init_weights)
    disc_audio_enc.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
video_rnn_posterior_optimizer = opt.optimizer(video_rnn_posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
video_rnn_prior_optimizer = opt.optimizer(video_rnn_prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
audio_rnn_optimizer = opt.optimizer(audio_rnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
audio_enc_optimizer = opt.optimizer(audio_enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
disc_optimizer = opt.optimizer(disc.parameters(), lr=opt.disc_lr, betas=(opt.beta1, 0.999))
disc_std_optimizer = opt.optimizer(disc_std.parameters(), lr=opt.disc_lr, betas=(opt.beta1, 0.999))
disc_encoder_optimizer = opt.optimizer(disc_encoder.parameters(), lr=opt.disc_lr, betas=(opt.beta1, 0.999))
disc_audio_enc_optimizer = opt.optimizer(disc_audio_enc.parameters(), lr=opt.disc_lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
classif_criterion = nn.CrossEntropyLoss()
mse_criterion = nn.MSELoss()


def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
    #   log( sqrt(
    #
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
    return kld.sum() / opt.batch_size


# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
video_rnn_posterior.cuda()
video_rnn_prior.cuda()
audio_rnn.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
audio_enc.cuda()
disc.cuda()
disc_std.cuda()
disc_encoder.cuda()
disc_audio_enc.cuda()
classif_criterion.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(
    train_data, num_workers=opt.data_threads, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory=True
)
test_loader = DataLoader(
    test_data, num_workers=opt.data_threads, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True
)


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = (
                utils.normalize_data(opt, dtype, sequence[0]),
                utils.normalize_data(opt, dtype, sequence[1]),
            )  # print(batch[1][1].size())
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = (utils.normalize_data(opt, dtype, sequence[0]), utils.normalize_data(opt, dtype, sequence[1]))
            yield batch


testing_batch_generator = get_testing_batch()


# --------- plotting funtions ------------------------------------
def plot(x, a, epoch):
    nsample = 20
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        h_list, h_target_list, audio_lst = [], [], []
        frame_predictor.hidden = frame_predictor.init_hidden()

        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i <= opt.n_past:  # Save the skip connection information
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            h_list.append(h)
            audio_lst.append(audio_enc(a[i])[0].detach())
            # Convert the lists into tensors
            h_list_tmp, audio_lst_tmp = torch.stack(h_list), torch.stack(audio_lst)
            # Processing for the seen frames
            if i < opt.n_past:
                h_target_list.append(encoder(x[i])[0].detach())
                h_target_list_tmp = torch.stack(h_target_list)
                # Forward pass through the transformers
                a_rep_rnn_lst = audio_rnn(audio_lst_tmp)
                h_target_posterior_lst = video_rnn_posterior(h_target_list_tmp)
                h_prior_lst = video_rnn_prior(h_list_tmp)
                # Forward pass through the LSTMs
                posterior(torch.cat([h_target_posterior_lst[-1], a_rep_rnn_lst[-1]], 1))  # z_t, _, _ =
                z_t_hat, _, _ = prior(torch.cat([h_prior_lst[-1], a_rep_rnn_lst[-1]], 1))
                frame_predictor(torch.cat([h, z_t_hat], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                a_rep_rnn_lst = audio_rnn(audio_lst_tmp)
                h_prior_lst = video_rnn_prior(h_list_tmp)
                z_t_hat, _, _ = prior(torch.cat([h_prior_lst[-1], a_rep_rnn_lst[-1]], 1))
                h = frame_predictor(torch.cat([h, z_t_hat], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot, min_mse_lst = [], []
    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse += mse_criterion(gen_seq[s][t][i], gt_seq[t][i]).data.cpu()
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        min_mse_lst.append(min_mse / opt.n_eval)
        s_list = [
            min_idx,
            np.random.randint(nsample),
            np.random.randint(nsample),
            np.random.randint(nsample),
            np.random.randint(nsample),
        ]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = "%s/gen/sample_%d.png" % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = "%s/gen/sample_%d.gif" % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)
    return min_mse_lst


def plot_rec(x, a, epoch):
    h_list, h_target_list, audio_lst = [], [], []
    frame_predictor.hidden = frame_predictor.init_hidden()

    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])

    for i in range(1, opt.n_past + opt.n_future):
        h = encoder(x[i - 1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i <= opt.n_past:  # Save the skip connection information
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        h_list.append(h)
        h_target_list.append(h_target)
        audio_lst.append(audio_enc(a[i])[0].detach())
        # Convert the lists to tensors
        h_list_tmp, h_target_list_tmp, audio_lst_tmp = (
            torch.stack(h_list),
            torch.stack(h_target_list),
            torch.stack(audio_lst),
        )
        # Start forward pass through the transformers now
        a_rep_rnn_lst = audio_rnn(audio_lst_tmp)
        h_target_posterior_lst = video_rnn_posterior(h_target_list_tmp)
        # Forward pass through LSTMs now
        z_t, _, _ = posterior(torch.cat([h_target_posterior_lst[-1], a_rep_rnn_lst[-1]], 1))
        # *z_t_lst.append(tmp[-1])
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past + opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = "%s/gen/rec_%d.png" % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


print("Dataloaders created and models pushed to GPU!!!")


def wt_sched_gen(opt, epoch=0):
    """Schedule for the weight of the generator loss"""
    if epoch <= 300:
        return opt.beta_gen_loss
    elif epoch <= 600:
        return 5.0 * opt.beta_gen_loss
    elif epoch <= 900:
        return 10.0 * opt.beta_gen_loss
    else:
        return 100.0 * opt.beta_gen_loss


# --------- training funtions ------------------------------------
def train(x, a, epoch):
    frame_predictor.zero_grad()
    video_rnn_posterior.zero_grad()
    video_rnn_prior.zero_grad()
    audio_rnn.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()
    audio_enc.zero_grad()
    disc.zero_grad()
    disc_std.zero_grad()
    disc_encoder.zero_grad()
    disc_audio_enc.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()

    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    disc_loss, gen_loss, real_loss, syn_loss, mismatch_loss, gen_syn_loss, gen_mismatch_loss = 0, 0, 0, 0, 0, 0, 0
    kld, mse = 0, 0
    h_list, h_target_list, audio_lst = [], [], []
    pred_vid_frame_buff, real_vid_frame_buff, audio_frame_buff = [], [], []
    for i in range(1, opt.n_past + opt.n_future):  # Stack the different feature encodings in a list
        h = encoder(x[i - 1])
        h_target_list.append(encoder(x[i])[0])
        if opt.last_frame_skip or i <= opt.n_past:  # Save the skip connection information
            h, skip = h
        else:
            h = h[0]
        h_list.append(h)
        audio_lst.append(audio_enc(a[i])[0])

        # Process the transformer list now
        # First convert the lists into stacks
        h_list_tmp, h_target_list_tmp, audio_lst_tmp = (
            torch.stack(h_list),
            torch.stack(h_target_list),
            torch.stack(audio_lst),
        )
        # Forward pass through transformers
        a_rep_rnn_lst = audio_rnn(audio_lst_tmp)
        h_target_posterior_lst = video_rnn_posterior(h_target_list_tmp)
        h_prior_lst = video_rnn_prior(h_list_tmp)
        # Forward pass through LSTMs now
        _, mu, logvar = posterior(torch.cat([h_target_posterior_lst[-1], a_rep_rnn_lst[-1]], 1))
        z_t_hat, mu_p, logvar_p = prior(torch.cat([h_prior_lst[-1], a_rep_rnn_lst[-1]], 1))
        h_pred = frame_predictor(torch.cat([h, z_t_hat], 1))

        # Now iterate over the responses step by step
        # Perform the decoding
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])  # Compute the MSE loss
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)  # Compute the KL loss

        # -------------------- Discriminator ------------------------ #
        # Forward pass through discriminator encoder
        x_hat_enc, _ = disc_encoder(x_pred)
        pred_vid_frame_buff.append(x_hat_enc)  # Append the current predicted frame to the end
        real_vid_frame_buff.append(disc_encoder(x[i])[0])  # Append the target video frame to the end
        audio_frame_buff.append(
            disc_audio_enc(a[i])[0]
        )  # Append the encoding of the current audio frame to the end # a_rep_rnn

        if len(pred_vid_frame_buff) > opt.disc_buffer:  # The buffer is full
            # *if len(pred_vid_frame_buff) > opt.disc_buffer: # We need to pop
            pred_vid_frame_buff, real_vid_frame_buff, audio_frame_buff = (
                pred_vid_frame_buff[1:],
                real_vid_frame_buff[1:],
                audio_frame_buff[1:],
            )

        elif (
            len(pred_vid_frame_buff) < opt.disc_buffer
        ):  # The buffer is full: # We don't compute gradients until the buffer is full capacity
            continue

        # Forward pass through the LSTM discriminator - with synthetic samples for discriminator
        disc.hidden = disc.init_hidden()
        # Iterate through the buffer
        for j in range(len(pred_vid_frame_buff)):
            if j == (opt.disc_buffer - 1):  # Executing the final frame of the buffer
                pred_prob_vec = disc(torch.cat([pred_vid_frame_buff[j].detach(), audio_frame_buff[j].detach()], 1))
            else:
                disc(torch.cat([real_vid_frame_buff[j].detach(), audio_frame_buff[j].detach()], 1))
        # Generate labels for discriminator
        l_z = torch.LongTensor(x[i - 1].size()[0])
        l_z.fill_(0)
        l_z = l_z.cuda()
        l_o = torch.LongTensor(x[i].size()[0])
        l_o.fill_(1)
        l_o = l_o.cuda()
        # Compute the classification loss for the final frame of the buffer
        syn_loss += classif_criterion(pred_prob_vec, Variable(l_z))
        # Forward pass through the std discriminator - with synthetic samples for discriminator
        std_pred_prob_vec = disc_std(x_hat_enc.detach())
        # Compute the classification loss for the final frame of the buffer
        syn_loss += classif_criterion(std_pred_prob_vec, Variable(l_z))

        # Forward pass through the discriminator - with real samples and matched sound
        disc.hidden = disc.init_hidden()
        # Iterate through the buffer
        for j in range(len(real_vid_frame_buff)):
            if j == (opt.disc_buffer - 1):  # Executing the final frame of the buffer
                real_prob_vec = disc(torch.cat([real_vid_frame_buff[j].detach(), audio_frame_buff[j].detach()], 1))
            else:
                disc(torch.cat([real_vid_frame_buff[j].detach(), audio_frame_buff[j].detach()], 1))
        # Compute the loss for the real samples
        real_loss += classif_criterion(real_prob_vec, Variable(l_o))
        # Forward pass through the std discriminator - with real samples for discriminator
        std_real_prob_vec = disc_std(real_vid_frame_buff[-1].detach())
        # Compute the loss for the real samples
        real_loss += classif_criterion(std_real_prob_vec, Variable(l_o))

        # Forward pass through the LSTM discriminator - with real samples but mismatched sound
        disc.hidden = disc.init_hidden()
        # Iterate through the buffer
        for j in range(len(real_vid_frame_buff)):
            if j == (opt.disc_buffer - 1):  # Executing the final frame of the buffer
                real_mismatch_prob_vec = disc(
                    torch.cat(
                        [
                            real_vid_frame_buff[j].detach(),
                            torch.cat([audio_frame_buff[j][1:], audio_frame_buff[j][0].unsqueeze(0)], 0).detach(),
                        ],
                        1,
                    )
                )
            else:
                disc(
                    torch.cat(
                        [
                            real_vid_frame_buff[j].detach(),
                            torch.cat([audio_frame_buff[j][1:], audio_frame_buff[j][0].unsqueeze(0)], 0).detach(),
                        ],
                        1,
                    )
                )
        # Compute the mismatch loss
        mismatch_loss += classif_criterion(real_mismatch_prob_vec, Variable(l_z))

        # Forward pass through the LSTM discriminator - with synthetic samples but mismatched sound
        disc.hidden = disc.init_hidden()
        # Iterate through the buffer
        for j in range(len(pred_vid_frame_buff)):
            if j == (opt.disc_buffer - 1):  # Executing the final frame of the buffer
                mismatch_prob_vec = disc(
                    torch.cat(
                        [
                            pred_vid_frame_buff[j].detach(),
                            torch.cat([audio_frame_buff[j][1:], audio_frame_buff[j][0].unsqueeze(0)], 0).detach(),
                        ],
                        1,
                    )
                )
            else:
                disc(
                    torch.cat(
                        [
                            real_vid_frame_buff[j].detach(),
                            torch.cat([audio_frame_buff[j][1:], audio_frame_buff[j][0].unsqueeze(0)], 0).detach(),
                        ],
                        1,
                    )
                )
        # Compute the mismatch loss
        mismatch_loss += classif_criterion(mismatch_prob_vec, Variable(l_z))

        # Forward pass through the discriminator - with synthetic samples for the generator
        disc.hidden = disc.init_hidden()
        # Iterate through the buffer
        for j in range(len(pred_vid_frame_buff)):
            if j == (opt.disc_buffer - 1):  # Executing the final frame of the buffer
                pred_prob_vec_gen = disc(torch.cat([pred_vid_frame_buff[j], audio_frame_buff[j]], 1))
            else:
                disc(torch.cat([real_vid_frame_buff[j].detach(), audio_frame_buff[j].detach()], 1))
        # Compute the generator loss: Minimize -log D(G(z))
        gen_loss += classif_criterion(pred_prob_vec_gen, Variable(l_o))
        # Forward pass through the std discriminator - with synthetic samples for the generator
        std_pred_prob_vec_gen = disc_std(x_hat_enc)

        # Compute the classification loss for the final frame of the buffer
        gen_syn_loss += classif_criterion(std_pred_prob_vec_gen, Variable(l_o))

        # Forward pass through the LSTM discriminator - with synthetic samples but mismatched sound for the generator
        disc.hidden = disc.init_hidden()
        # Iterate through the buffer
        for j in range(len(pred_vid_frame_buff)):
            if j == (opt.disc_buffer - 1):  # Executing the final frame of the buffer
                mismatch_prob_vec_gen = disc(
                    torch.cat(
                        [
                            pred_vid_frame_buff[j],
                            torch.cat([audio_frame_buff[j][1:], audio_frame_buff[j][0].unsqueeze(0)], 0),
                        ],
                        1,
                    )
                )
            else:
                disc(
                    torch.cat(
                        [
                            real_vid_frame_buff[j].detach(),
                            torch.cat([audio_frame_buff[j][1:], audio_frame_buff[j][0].unsqueeze(0)], 0).detach(),
                        ],
                        1,
                    )
                )
        # Compute the mismatch loss
        gen_mismatch_loss += classif_criterion(mismatch_prob_vec_gen, Variable(l_o))

    # Compute the total generator loss and backprop, use the disc loss only after disc warm start
    if epoch > 0:
        loss = (
            mse + (kld * opt.beta) + (((gen_loss + gen_syn_loss + gen_mismatch_loss) / 3.0) * wt_sched_gen(opt, epoch))
        )  #
    else:
        loss = mse + (kld * opt.beta)

    loss.backward()

    frame_predictor_optimizer.step()
    video_rnn_posterior_optimizer.step()
    video_rnn_prior_optimizer.step()
    audio_rnn_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    audio_enc_optimizer.step()

    # Reset the discriminator gradient to 0
    disc.zero_grad()
    disc_std.zero_grad()
    disc_encoder.zero_grad()
    disc_audio_enc.zero_grad()
    # Compute the total discriminator loss and backprop
    disc_loss += (real_loss / 2.0) + ((syn_loss + mismatch_loss) / 8.0)
    disc_loss.backward()
    disc_optimizer.step()  # Update only the discriminator components
    disc_std_optimizer.step()
    disc_encoder_optimizer.step()
    disc_audio_enc_optimizer.step()

    return (
        disc_loss.data.cpu().numpy() / (opt.n_past + opt.n_future),
        syn_loss.data.cpu().numpy() / (opt.n_past + opt.n_future),
        kld.data.cpu().numpy() / (opt.n_future + opt.n_past),
        mse.data.cpu().numpy() / (opt.n_future + opt.n_past),
    )


# --------- training loop ------------------------------------
disc_loss_list, gen_loss_list, mse_loss_list, kl_loss_list, train_mse_list, prev_mse = [], [], [], [], [], 1e5
# Generate sample batch from test set
x_test, a_test = next(testing_batch_generator)

for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    audio_enc.train()
    video_rnn_posterior.train()
    video_rnn_prior.train()
    audio_rnn.train()
    disc.train()
    disc_std.train()
    disc_encoder.train()
    disc_audio_enc.train()

    epoch_disc = 0
    epoch_gen = 0
    epoch_kld = 0
    epoch_mse_train = 0
    progress = progressbar.ProgressBar(
        max_value=int(len(os.listdir(os.path.join(opt.data_root, "train", "gif"))) / opt.batch_size)
    ).start()
    for i in range(
        int(len(os.listdir(os.path.join(opt.data_root, "train", "gif"))) / opt.batch_size)
    ):  # range(opt.epoch_size):
        progress.update(i + 1)
        x, a = next(training_batch_generator)  # , _
        # train frame_predictor
        disc_loss, gen_loss, kld, mse = train(x, a, epoch)  #
        epoch_disc += disc_loss
        epoch_gen += gen_loss
        epoch_kld += kld
        epoch_mse_train += mse
        # *if i==10: break

    progress.finish()
    utils.clear_progressbar()

    # plot some stuff
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()
    audio_enc.eval()
    video_rnn_posterior.eval()
    video_rnn_prior.eval()
    audio_rnn.eval()

    # *x, a = next(testing_batch_generator) # , _
    epoch_mse = plot(x_test, a_test, epoch)
    plot_rec(x_test, a_test, epoch)
    print(
        "[%02d] mse loss: %.5f | disc loss: %.5f | gen loss: %.5f | kld loss: %.5f | train mse loss: %.5f (%d)"
        % (
            epoch,
            sum(epoch_mse) * 1.0 / len(epoch_mse),
            epoch_disc / (i + 1),
            epoch_gen / (i + 1),
            epoch_kld / (i + 1),
            epoch_mse_train / (i + 1),
            epoch * (i + 1) * opt.batch_size,
        )
    )

    if prev_mse > (sum(epoch_mse) * 1.0 / len(epoch_mse)):
        # save the model
        torch.save(
            {
                "encoder": encoder,
                "decoder": decoder,
                "frame_predictor": frame_predictor,
                "video_rnn_posterior": video_rnn_posterior,
                "video_rnn_prior": video_rnn_prior,
                "audio_rnn": audio_rnn,
                "posterior": posterior,
                "prior": prior,
                "audio_enc": audio_enc,
                "disc": disc,
                "disc_std": disc_std,
                "disc_encoder": disc_encoder,
                "disc_audio_enc": disc_audio_enc,
                "loss": sum(epoch_mse) * 1.0 / len(epoch_mse),
                "opt": opt,
            },
            "%s/model.pth" % opt.log_dir,
        )

        # save the optimizer
        torch.save(
            {
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "frame_predictor_optimizer": frame_predictor_optimizer.state_dict(),
                "video_rnn_posterior_optimizer": video_rnn_posterior_optimizer.state_dict(),
                "video_rnn_prior_optimizer": video_rnn_prior_optimizer.state_dict(),
                "audio_rnn_optimizer": audio_rnn_optimizer.state_dict(),
                "posterior_optimizer": posterior_optimizer.state_dict(),
                "prior_optimizer": prior_optimizer.state_dict(),
                "audio_enc_optimizer": audio_enc_optimizer.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(),
                "disc_std_optimizer": disc_std_optimizer.state_dict(),
                "disc_encoder_optimizer": disc_encoder_optimizer.state_dict(),
                "disc_audio_enc_optimizer": disc_audio_enc_optimizer.state_dict(),
            },
            "%s/opt.pth" % opt.log_dir,
        )

        prev_mse = sum(epoch_mse) * 1.0 / len(epoch_mse)

    mse_loss_list.append(sum(epoch_mse) * 1.0 / len(epoch_mse))
    kl_loss_list.append(epoch_kld / (i + 1))
    # Log the discriminator and generator losses
    disc_loss_list.append(epoch_disc / (i + 1))
    gen_loss_list.append(epoch_gen / (i + 1))
    train_mse_list.append(epoch_mse_train / (i + 1))

    if epoch % 10 == 0:
        with open(os.path.join(opt.log_dir, "plots", "Model_perf.pickle"), "wb") as f:
            pickle.dump([mse_loss_list, disc_loss_list, gen_loss_list, kl_loss_list, train_mse_list], f)

        # save the model
        torch.save(
            {
                "encoder": encoder,
                "decoder": decoder,
                "frame_predictor": frame_predictor,
                "video_rnn_posterior": video_rnn_posterior,
                "video_rnn_prior": video_rnn_prior,
                "audio_rnn": audio_rnn,
                "posterior": posterior,
                "prior": prior,
                "audio_enc": audio_enc,
                "disc": disc,
                "disc_std": disc_std,
                "disc_encoder": disc_encoder,
                "disc_audio_enc": disc_audio_enc,
                "loss": sum(epoch_mse) * 1.0 / len(epoch_mse),
                "opt": opt,
            },
            "%s/model_%d.pth" % (opt.log_dir, epoch),
        )

        # save the optimizer
        torch.save(
            {
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "frame_predictor_optimizer": frame_predictor_optimizer.state_dict(),
                "video_rnn_posterior_optimizer": video_rnn_posterior_optimizer.state_dict(),
                "video_rnn_prior_optimizer": video_rnn_prior_optimizer.state_dict(),
                "audio_rnn_optimizer": audio_rnn_optimizer.state_dict(),
                "posterior_optimizer": posterior_optimizer.state_dict(),
                "prior_optimizer": prior_optimizer.state_dict(),
                "audio_enc_optimizer": audio_enc_optimizer.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(),
                "disc_std_optimizer": disc_std_optimizer.state_dict(),
                "disc_encoder_optimizer": disc_encoder_optimizer.state_dict(),
                "disc_audio_enc_optimizer": disc_audio_enc_optimizer.state_dict(),
            },
            "%s/opt_%d.pth" % (opt.log_dir, epoch),
        )
