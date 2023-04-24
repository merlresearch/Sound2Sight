# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pickle

import numpy as np
from PIL import Image


class MovingMNIST_STFT_Resize(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=1, image_size=48, deterministic=False):
        self.path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1
        self.train = train
        if train:
            self.spec_data = os.listdir(os.path.join(self.path, "train", "stft_wav_pickle"))
            self.gif_data = os.listdir(os.path.join(self.path, "train", "gif"))
        else:
            self.spec_data = os.listdir(os.path.join(self.path, "test", "stft_wav_pickle"))
            self.gif_data = os.listdir(os.path.join(self.path, "test", "gif"))

        self.N = len(self.gif_data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # self.id_lst.append(index); print('Index is: ' + str(len(self.id_lst)))
        self.set_seed(index)
        image_size = self.image_size
        x = np.zeros((self.seq_len, image_size, image_size, self.channels), dtype=np.float32)

        # Load the index-th file into x and
        if self.train:
            # Read the audio file
            with open(os.path.join(self.path, "train", "stft_wav_pickle", str(self.spec_data[index])), "rb") as f:
                x_a = pickle.load(f)

            img = Image.open(os.path.join(self.path, "train", "gif", str(self.spec_data[index][:-7]) + ".gif"))
        else:
            # Read the audio file
            with open(os.path.join(self.path, "test", "stft_wav_pickle", str(self.spec_data[index])), "rb") as f:
                x_a = pickle.load(f)

            img = Image.open(os.path.join(self.path, "test", "gif", str(self.spec_data[index][:-7]) + ".gif"))

        for _ in range(self.seq_len):
            img.seek(_)
            img_1 = img.resize(
                (image_size, image_size), Image.ANTIALIAS
            )  # Since the input gifs are not in the resized shape
            tmp = np.array(img_1)
            frm = tmp / np.amax(tmp)
            x[_, :, :, 0] = frm

        # Expand the dimension of the spctrogram array
        x_a = np.expand_dims(x_a, axis=3)[: self.seq_len, :, :, :]
        # *print('Shape of x_a is: ' + str(x_a.shape))

        return x, x_a, self.spec_data[index][:-7]


class MovingMNIST_Resize(object):

    """Data Handler that reads M3S0 dataset with MFCC Features."""

    def __init__(self, train, data_root, seq_len=20, num_digits=1, image_size=48, deterministic=False):
        self.path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1
        self.train = train

        if train:
            self.spec_data = os.listdir(os.path.join(self.path, "train", "wav_pickle"))
            self.gif_data = os.listdir(os.path.join(self.path, "train", "gif"))
        else:
            self.spec_data = os.listdir(os.path.join(self.path, "test", "wav_pickle"))
            self.gif_data = os.listdir(os.path.join(self.path, "test", "gif"))

        self.N = len(self.gif_data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # self.id_lst.append(index); print('Index is: ' + str(len(self.id_lst)))
        self.set_seed(index)
        image_size = self.image_size
        x = np.zeros((self.seq_len, image_size, image_size, self.channels), dtype=np.float32)
        pos_lst = []
        # Load the index-th file into x and
        if self.train:
            # Read the audio file
            with open(os.path.join(self.path, "train", "wav_pickle", str(self.spec_data[index])), "rb") as f:
                x_a = pickle.load(f)

            img = Image.open(os.path.join(self.path, "train", "gif", str(self.spec_data[index][:-7]) + ".gif"))
        else:
            # Read the audio file
            with open(os.path.join(self.path, "test", "wav_pickle", str(self.spec_data[index])), "rb") as f:
                x_a = pickle.load(f)

            img = Image.open(os.path.join(self.path, "test", "gif", str(self.spec_data[index][:-7]) + ".gif"))

        for _ in range(self.seq_len):
            img.seek(_)
            img_1 = img.resize(
                (image_size, image_size), Image.ANTIALIAS
            )  # Since the input gifs are not in the resized shape
            tmp = np.array(img_1)
            frm = tmp / np.amax(tmp)
            x[_, :, :, 0] = frm

        # Expand the dimension of the spctrogram array
        x_a = np.expand_dims(x_a, axis=3)[: self.seq_len, :, :, :]

        return x, x_a, self.spec_data[index][:-7]  # *, pos_lst
