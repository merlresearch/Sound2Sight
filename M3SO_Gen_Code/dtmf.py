#!/usr/bin/env python3

# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2003 Jiwon Hahn
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: GPL-3.0-or-later


from math import *
from sys import *
from wave import *

import scipy.io.wavfile

PI2 = 6.283185306
scale = 32767.0  # 16-bit unsigned short
FR = 4400  # framerate. 44000
FPS = 10  # 20 #5 #10

keys = "1", "2", "3", "A", "4", "5", "6", "B", "7", "8", "9", "C", "*", "0", "#", "D"
F1 = [16, 32, 64, 128]  # [697,770,852,941]
F2 = [256, 512, 1024, 2048]  # [1209, 1336, 1477, 1633]


def encoder(symbol):
    for i in range(16):
        if symbol == keys[i]:
            f1 = F1[i / 4]  # row
            f2 = F2[i % 4]  # column
    data = range(FR)
    for i in range(FR):
        p = i * 1.0 / FR
        data[i] = int(scale + (sin(p * f1 * PI2) + sin(p * f2 * PI2)) / 2 * scale)
    store_wav(data, symbol)


# endian inversion for unsigned 8 bit
def inv_endian(num):
    b = num2bit(num)
    N = len(b)
    sum = 0
    for i in range(N):
        sum += int(b.pop(0)) * 2**i
    return sum


def num2bit(num):  # 8bit
    b = []
    for i in range(7, -1, -1):
        if num >= 2**i:
            b.append("1")
            num -= 2**i
        else:
            b.append("0")
    return b


def store_wav(data, filename):
    scipy.io.wavfile.write(filename, FR, data)
    w = open(filename, "r")
    print("Number of frames in file: " + str(w.getnframes()))


def read_wav(symbol):
    fin = open("p" + symbol + ".wav", "r")
    n = fin.getnframes()
    d = fin.readframes(n)
    fin.close()

    data = []
    for i in range(n):
        LS8bit, MS8bit = ord(d[2 * i]), ord(d[2 * i + 1])
        data.append((MS8bit << 8) + LS8bit)
    return data


# Decoder takes a DTMF signal file (.wav), sampled at 44,000
# 16-bit samples per second, and decode the corresponding symbol X.


def decoder(X):
    data = read_wav(X)
    temp = []
    for f1 in F1:
        for f2 in F2:
            diff = 0
            for i in range(FR):  # assume phase has not shifted dramatically
                p = i * 1.0 / FR
                S = int(scale + scale * (sin(p * f1 * PI2) + sin(p * f2 * PI2)) / 2)
                diff += abs(S - data[i])
            temp.append((diff, f1, f2))
    f1, f2 = min(temp)[1:]  # retrieve the frequency of minimum signal distortion
    i, j = F1.index(f1), F2.index(f2)
    X = keys[4 * i + j]
    print("Decoded key is: {}".format(X))
    return X


def menu():
    while 1:
        print("**************************")
        print("1\t2\t3\tA")
        print("4\t5\t6\tB")
        print("7\t8\t9\tC")
        print("*\t0\t#\tD")
        print("**************************")
        X = raw_input("Enter a key, or x to exit: ")
        if X not in keys:
            if X is "x":
                exit(0)
            print("Invalid key...")
        else:
            return X
