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
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils3_48 as utils

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--data_root", default="data", help="root directory for data")
parser.add_argument("--model_path", default="", help="path to model")
parser.add_argument("--log_dir", default="", help="directory to save generations to")
parser.add_argument("--seed", default=1, type=int, help="manual seed")
parser.add_argument("--n_past", type=int, default=30, help="number of frames to condition on")
parser.add_argument("--n_future", type=int, default=60, help="number of frames to predict")
parser.add_argument("--num_threads", type=int, default=0, help="number of data loading threads")
parser.add_argument("--nsample", type=int, default=100, help="number of samples")
parser.add_argument("--N", type=int, default=1000, help="number of samples")
parser.add_argument("--model_epoch", type=int, help="epoch number of the training model")
parser.add_argument("--log_path", help="path where the log file is to be stored")

opt = parser.parse_args()
os.makedirs("%s" % opt.log_dir, exist_ok=True)

opt.n_eval = opt.n_past + opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
frame_predictor = tmp["frame_predictor"]
posterior = tmp["posterior"]
prior = tmp["prior"]
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp["encoder"]
decoder = tmp["decoder"]
video_rnn_posterior = tmp["video_rnn_posterior"]
video_rnn_prior = tmp["video_rnn_prior"]
audio_rnn = tmp["audio_rnn"]
audio_enc = tmp["audio_enc"]
encoder.eval()
decoder.eval()
video_rnn_posterior.eval()
video_rnn_prior.eval()
audio_rnn.eval()
audio_enc.eval()

frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
video_rnn_posterior.batch_size = opt.batch_size
video_rnn_prior.batch_size = opt.batch_size
audio_rnn.batch_size = opt.batch_size
opt.g_dim = tmp["opt"].g_dim
opt.z_dim = tmp["opt"].z_dim
opt.num_digits = tmp["opt"].num_digits

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
video_rnn_posterior.cuda()
video_rnn_prior.cuda()
audio_rnn.cuda()
audio_enc.cuda()

# ---------------- set the options ----------------
opt.dataset = tmp["opt"].dataset
opt.last_frame_skip = tmp["opt"].last_frame_skip
opt.channels = tmp["opt"].channels
opt.image_width = tmp["opt"].image_width
opt.log_dir_train = tmp["opt"].log_dir

print(opt)
print(video_rnn_posterior)
# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(
    train_data, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory=True
)
test_loader = DataLoader(
    test_data, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True
)


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = (utils.normalize_data(opt, dtype, sequence[0]), utils.normalize_data(opt, dtype, sequence[1]))
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = (utils.normalize_data(opt, dtype, sequence[0]), utils.normalize_data(opt, dtype, sequence[1]))
            yield batch


testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------


def make_gifs(x, a, idx, name):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    h_list, h_target_list, audio_lst, posterior_gen = [], [], [], []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i <= opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        # Append the encodings to a list
        h_list.append(h)
        h_target_list.append(h_target)
        # Encode the audio
        audio_lst.append(audio_enc(a[i])[0].detach())  # *a_rep, _ = audio_enc(a[i])
        # Convert the lists to tensors
        h_list_tmp, h_target_list_tmp, audio_lst_tmp = (
            torch.stack(h_list),
            torch.stack(h_target_list),
            torch.stack(audio_lst),
        )
        # Start forward pass through the transformers now
        a_rep_rnn_lst = audio_rnn(audio_lst_tmp)  # a_rep_rnn = audio_rnn(a_rep)
        h_target_posterior_lst = video_rnn_posterior(
            h_target_list_tmp
        )  # h_target_posterior = video_rnn_posterior(h_target)
        # Forward pass through LSTMs now
        z_t, _, _ = posterior(
            torch.cat([h_target_posterior_lst[-1], a_rep_rnn_lst[-1]], 1)
        )  # take the mean #_, z_t, _= posterior(torch.cat([h_target_posterior, a_rep_rnn], 1))

        # If we are in the past
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))  # frame_predictor(torch.cat([h, z_t], 1))
            posterior_gen.append(x[i])
            x_in = x[i]
        else:  # For future frame representations
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)  # * """

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    # Generate nsample sample for every x
    for s in range(nsample):
        progress.update(s + 1)
        gen_seq = []
        gt_seq = []
        h_list, h_target_list, audio_lst = [], [], []
        # Initialize the LSTMs
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        # Iterate over all frames
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i <= opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            h_list.append(h)
            audio_lst.append(audio_enc(a[i])[0].detach())
            # Convert the lists into tensors
            h_list_tmp, audio_lst_tmp = torch.stack(h_list), torch.stack(audio_lst)
            # If in the past
            if i < opt.n_past:
                h_target_list.append(encoder(x[i])[0].detach())
                h_target_list_tmp = torch.stack(h_target_list)
                # Forward pass through the transformers
                a_rep_rnn_lst = audio_rnn(audio_lst_tmp)
                h_target_posterior_lst = video_rnn_posterior(h_target_list_tmp)
                h_prior_lst = video_rnn_prior(h_list_tmp)
                # Forward pass through the LSTMs
                posterior(torch.cat([h_target_posterior_lst[-1], a_rep_rnn_lst[-1]], 1))
                z_t_hat, _, _ = prior(torch.cat([h_prior_lst[-1], a_rep_rnn_lst[-1]], 1))
                frame_predictor(torch.cat([h, z_t_hat], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                a_rep_rnn_lst = audio_rnn(audio_lst_tmp)
                h_prior_lst = video_rnn_prior(h_list_tmp)
                z_t_hat, _, _ = prior(torch.cat([h_prior_lst[-1], a_rep_rnn_lst[-1]], 1))
                h = frame_predictor(torch.cat([h, z_t_hat], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    progress.finish()
    utils.clear_progressbar()

    # """###### ssim based sort ######
    for i in range(opt.batch_size):
        gifs = [[] for t in range(opt.n_eval)]
        text = [[] for t in range(opt.n_eval)]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt
            gifs[t].append(add_border(x[t][i], "green"))
            text[t].append("Ground\ntruth")
            # posterior
            if t < opt.n_past:
                color = "green"
            else:
                color = "red"
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append("Approx.\nposterior")
            # best
            if t < opt.n_past:
                color = "green"
            else:
                color = "red"
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append("Best SSIM")
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append("Random\nsample %d" % (s + 1))

        fname = "%s/%s_%d.gif" % (opt.log_dir, name, idx + i)
        utils.save_gif_with_text(fname, gifs, text)

        # Save the frames as a new list
        frm_lst = [all_gen[sidx][_][i].data.cpu().numpy() for _ in range(len(all_gen[sidx]))]
        # Save the gif corresponding to the best ssim for further evaluation
        with open(os.path.join(opt.log_dir, name + "_generations", str(idx + i) + ".pickle"), "wb") as f:
            pickle.dump(frm_lst, f)  # * """

    return np.amax(ssim, axis=1), np.amax(psnr, axis=1)


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w + 2 * pad + 30, w + 2 * pad))
    if color == "red":
        px[0] = 0.7
    elif color == "green":
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad : w + pad, pad : w + pad] = x
    else:
        px[:, pad : w + pad, pad : w + pad] = x
    return px


ssim_val, psnr_val = [], []

for i in range(0, opt.N, opt.batch_size):
    # Get the test set data
    test_x, test_a = next(testing_batch_generator)

    # Run the evaluation for the test set
    s_val, p_val = make_gifs(test_x, test_a, i, "test")
    ssim_val.append(s_val)
    psnr_val.append(p_val)
    print("Processed Batch: " + str(i))  # ; break;

ssim_mean = np.mean(np.concatenate(ssim_val, axis=0), axis=0)
psnr_mean = np.mean(np.concatenate(psnr_val, axis=0), axis=0)

print("The mean SSIM is: " + str(ssim_mean))
print("The mean PSNR is: " + str(psnr_mean))

with open(os.path.join(opt.log_path, "plots", "Model_perf_test_" + str(opt.model_epoch) + "_max.pickle"), "wb") as f:
    pickle.dump([ssim_mean, psnr_mean], f)
