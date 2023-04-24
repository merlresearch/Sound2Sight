#!/usr/bin/env python3

# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import os
import pickle
import sys

import dtmf
import numpy as np
from PIL import Image

###########################################################################################
# This script extends the moving mnist of Srivastava et al., by adding a block of given size to
# an arbitrary location in the canvas after a predefined time. We also add audio to the movement
# of the digits, each digit with its own distinct frequency, while its amplitude varies with distance
# from the origin. When the digits hit the blocks, we hear an amplitude double that of the original digit,
# with some added offset as well.
###########################################################################################

key = np.random.randint(1000)
print(key)
np.random.seed(key)
FR = dtmf.FR  # snd frames per sec.
FPS = dtmf.FPS  # images frame per sec


# helper functions
def arr_from_img(im, mean=0, std=1):
    """
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract

    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    """
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255.0 - mean) / std


def get_image_from_array(X, index, mean=0, std=1):
    """
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    """
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.0) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


# loads mnist from web on demand
def load_dataset(training=True, dest=""):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source="http://yann.lecun.com/exdb/mnist/"):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        import gzip

        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    if training:
        return load_mnist_images(os.path.join(dest, "train-images-idx3-ubyte.gz")), load_mnist_labels(
            os.path.join(dest, "train-labels-idx1-ubyte.gz")
        )
    return load_mnist_images(os.path.join(dest, "t10k-images-idx3-ubyte.gz")), load_mnist_labels(
        os.path.join(dest, "train-labels-idx1-ubyte.gz")
    )


# Encoder takes a symbol X as input and generate a
# corresponding one second long DTMF tone, sampled at
# 44,000 16-bit samples/sec, and store it in a wav file.


def generate_audio(symbol, amplitude, bnd, blk):  # duration is the time (in sec) for which the sound will be.
    if bnd and not blk:
        symbol = 11  # sound of bounce from a boundary.
    elif not bnd and blk:
        symbol = 12  # sound of bounce from the block.

    f1 = dtmf.F1[int(symbol / 4.0)]  # row
    f2 = dtmf.F2[symbol % 4]  # column
    num_snd_samples = int(float(FR) / FPS)
    data = np.zeros((num_snd_samples, 1), dtype="int")
    for i in range(num_snd_samples):
        p = i * 1.0 / num_snd_samples
        data[i] = dtmf.scale + amplitude * ((np.sin(p * f1 * dtmf.PI2) + np.sin(p * f2 * dtmf.PI2)) / 2) * dtmf.scale
    return data.transpose(1, 0)


def coord_in(lims, pt):
    return pt[0] >= lims[0] and pt[0] <= lims[2] and pt[1] >= lims[1] and pt[1] <= lims[3]


def is_intersect(blk, dig):
    # check if any corner of r1 is inside the rectangle r2.
    coords = [(dig[0], dig[1]), (dig[2], dig[1]), (dig[0], dig[3]), (dig[2], dig[3])]
    intersect = [coord_in(blk, r) for r in coords]
    return any(intersect), coords[intersect == True]


def make_box(pos, sz):
    return (pos[0], pos[1], pos[0] + sz, pos[1] + sz)


def cycle_coords(box):
    return [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]


# blk is the block coords, dig = dig coords, pos is the previous pos of the dig.
# see which face of the block did the line segment p2 - p1 intersect with? The key idea is that
# such an intersection pt will be between the two end points of the two line segments. So given p1, p2 and x1, x2 as the two line segments,
# find alpha, beta such that p1*alpha + p2*(1-alpha) = x1 * beta + x2 * (1-beta), and alpha, beta > 0. The pts, x1,x2 of the block
# that satisfies the positivity condition will be the face that the digits intersect with.
def get_bounce_dir(blk, dig, pos):
    # check if any corner of r1 is inside the rectangle r2.
    p1_coords, p2_coords, blk_coords = cycle_coords(pos), cycle_coords(dig), cycle_coords(blk)
    intersect = [coord_in(blk, r) for r in p2_coords]
    p1 = np.array(p1_coords[intersect == True])
    p2 = np.array(p2_coords[intersect == True])

    p12 = p1 - p2
    b = np.array(blk_coords)
    blk_faces = [b[0] - b[1], b[1] - b[2], b[2] - b[3], b[3] - b[0]]
    sol = [
        np.matmul(np.linalg.inv(np.stack([p12, -blk_faces[t]]).transpose()), p2 - b[(t + 1) % 4]) for t in range(4)
    ]  # solve the linear eqn.
    sol = [
        sol[t][0] > 0 and sol[t][1] > 0 for t in range(4)
    ]  # solution will provide the intercepts, that lie between the two end points of the two line segs.
    if sum(sol) > 1:  # if we have intersection with more than one side, just reverse the velocity.
        vel = [-1.0, -1.0]
    elif sum(sol) == 1:
        if sol[0] or sol[2]:  # if its an intersection with a vertical face, then reverse that velocity.
            vel = [1.0, -1.0]
        else:
            vel = [-1.0, 1.0]  # if its an intrsect with horiz face, then reverse taht velocity.
    else:
        # we do not have a good solution! Any movement should be fine.
        vel = [-1.0, -1.0]
    return vel


def get_block_position(positions, blk_size, digit_size, canvas_size):
    digit_rects = [make_box(p, digit_size) for p in positions]
    x_lim, y_lim = 0.75 * canvas_size[0] - blk_size, 0.75 * canvas_size[1] - blk_size
    try_limit = 10000
    while try_limit > 0:
        bp = np.asarray((np.random.rand() * x_lim + canvas_size[0] / 4, np.random.rand() * y_lim + canvas_size[1] / 4))
        blk_rect = make_box(bp, blk_size)  # (bp[0], bp[1], bp[0] + blk_size, bp[1] + blk_size)
        if any([is_intersect(blk_rect, drect)[0] for drect in digit_rects]):
            try_limit -= 1
            continue
        else:
            break
    if try_limit <= 0:
        print("Error: Could not find a block location that will not overlap with the digits... continuing...")
    return bp


def generate_moving_mnist(
    training,
    shape=(64, 64),
    num_frames=30,
    num_images=100,
    original_size=28,
    nums_per_image=2,
    block_start_frame=10,
    dest="",
):
    """
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_images: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.
        block_start_frame: is the frame when the block is introduced. Use something beyond 1.

    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height
    """
    mnist, labels = load_dataset(training, dest)
    assert len(mnist) == len(labels)
    width, height = shape
    assert block_start_frame > 1 and block_start_frame < num_frames

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    img_dataset = np.empty((num_images, num_frames, 1, width, height), dtype=np.uint8)
    snd_dataset = np.empty(
        (num_images, num_frames, int(float(FR) / FPS)), dtype=np.int16
    )  # num snd smaples per frame. # np.uint16
    blck_pos_lst = []
    print("Shape of snd_dataset: " + str(snd_dataset.shape))

    for img_idx in range(num_images):  # this is the num of sequences.
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_image) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
        # Get a list containing two PIL images randomly sampled from the database
        rand_lst = np.random.randint(0, mnist.shape[0], nums_per_image)
        mnist_images = [
            Image.fromarray(get_image_from_array(mnist, r, mean=0)).resize(
                (original_size, original_size), Image.ANTIALIAS
            )
            for r in rand_lst
        ]
        mnist_labels = [labels[r] for r in rand_lst]
        print("Video Number: " + str(img_idx) + " Digit: " + str(mnist_labels))
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)])

        # *samp_orig = []
        block_added = False
        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):

            canvases = [Image.new("L", (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            if frame_idx > block_start_frame:
                if not block_added:
                    # find a location for the block that does not overlap with the digits.
                    bs = original_size + 4  # block size
                    block_added = True
                    bp = get_block_position(positions, bs, original_size, (width, height))  # block_position
                    bp = bp.astype("int")
                    blk_lims = (bp[0], bp[1], bp[0] + bs, bp[1] + bs)
                    blck_pos_lst.append([bp[0], bp[1], bs])
                canvas[
                    0, bp[0] : bp[0] + bs, bp[1] : bp[1] + bs
                ] = 1  # Since the location of the block does not change over time

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            boundary_bounce, block_bounce = False, False
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1 :]))
                        boundary_bounce = True

                if block_added:
                    dig = make_box(pos, original_size)
                    if is_intersect(blk_lims, dig)[0]:
                        # find which face of blk_dims did dig intersect with.
                        for j, sgn in enumerate(get_bounce_dir(blk_lims, dig, make_box(positions[i], original_size))):
                            veloc[i] = list(list(veloc[i][:j]) + [sgn * veloc[i][j]] + list(veloc[i][j + 1 :]))
                            block_bounce = True

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            img_dataset[img_idx, frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)
            snd = np.zeros((1, snd_dataset.shape[2]), dtype=np.float)
            for t in range(nums_per_image):
                snd += generate_audio(
                    mnist_labels[t],
                    np.linalg.norm(positions[t]) / (np.sqrt(np.sum((np.array(shape) - original_size) ** 2.0))),
                    boundary_bounce,
                    block_bounce,
                )
            snd_dataset[img_idx, frame_idx] = (snd / nums_per_image).astype("int")

    return img_dataset, snd_dataset, blck_pos_lst


def main(
    training,
    dest,
    filetype="npz",
    frame_size=64,
    num_frames=30,
    num_images=100,
    original_size=28,
    nums_per_image=2,
    block_start_frame=10,
):
    dat, snd, blck_lst = generate_moving_mnist(
        training,
        shape=(frame_size, frame_size),
        num_frames=num_frames,
        num_images=num_images,
        original_size=original_size,
        nums_per_image=nums_per_image,
        block_start_frame=block_start_frame,
        dest=dest,
    )
    if filetype == "npz":
        np.savez(dest, dat)
    elif filetype == "jpg":
        for i in range(dat.shape[0]):
            Image.fromarray(get_image_from_array(dat, i, mean=0)).save(os.path.join(dest, "{}.jpg".format(i)))
    elif filetype == "gif":
        import imageio

        if not os.path.exists(dest):
            [os.mkdir(t) for t in [dest, dest + "/gif/"]]

        for t in dat.shape[0]:
            images = []
            for i in range(dat[t].shape[0]):
                img = Image.fromarray(get_image_from_array(dat[t], i, mean=0))
                images.append(img)
            imageio.mimsave(os.path.join(dest + "/gif/" + str(t) + ".gif"), images)
    elif filetype == "mp4":
        import imageio

        if not os.path.exists(dest):
            [os.mkdir(t) for t in [dest, dest + "/gif/", dest + "/wav/", dest + "/avi/", dest + "/block_pos/"]]

        for t in range(dat.shape[0]):
            images = []
            for i in range(dat[t].shape[0]):
                img = Image.fromarray(get_image_from_array(dat[t], i, mean=0))
                images.append(img)
            imageio.mimsave(os.path.join(dest + "/gif/" + str(t) + ".gif"), images)
            dtmf.store_wav(snd[t].reshape(-1), dest + "/wav/" + str(t) + ".wav")

            os.system(
                "ffmpeg -loglevel quiet -i "
                + dest
                + "/gif/"
                + str(t)
                + ".gif -i "
                + dest
                + "/wav/"
                + str(t)
                + ".wav -vcodec mpeg4 -r "
                + str(FPS)
                + " -y ./"
                + dest
                + "/avi/"
                + str(t)
                + ".avi"
            )

            with open(os.path.join(dest, "block_pos", str(t) + ".pickle"), "wb") as f:
                pickle.dump(blck_lst[t], f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Command line options")
    parser.add_argument("--dest", type=str, dest="dest", default="sample_gen")
    parser.add_argument("--filetype", type=str, dest="filetype", default="mp4")
    parser.add_argument("--training", type=bool, dest="training", default=True)
    parser.add_argument("--frame_size", type=int, dest="frame_size", default=48)
    parser.add_argument("--num_frames", type=int, dest="num_frames", default=30)  # length of each sequence
    parser.add_argument("--num_images", type=int, dest="num_images", default=20)  # number of sequences to generate
    parser.add_argument(
        "--original_size", type=int, dest="original_size", default=28
    )  # size of mnist digit within frame
    parser.add_argument(
        "--nums_per_image", type=int, dest="nums_per_image", default=1
    )  # number of digits in each frame
    parser.add_argument("--block_start_frame", type=int, dest="block_start_frame", default=10)
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
