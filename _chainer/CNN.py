# -*- Coding: utf-8 -*-
# Packages
import collections
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable

# # torch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# --- unused packages ---
# import os
# import glob
# import pickle
# import chainer
# import numpy as np
# from chainer import cuda
# from PIL import Image

# MyPackages
import GlobalVariable as gv


class CNN(Chain):
# class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=None, out_channels=96, padding=3, stride=1, kernel_size=7).to('cuda')

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, padding=3, stride=1, kernel_size=7, groups=)
        # self.bn1 = nn.BatchNorm2d(96)
        # self.conv2 = nn.Conv2d()
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d()
        # self.conv4 = nn.Conv2d()
        # self.conv5 = nn.Conv2d()
        # self.fc6 = nn.Linear()
        # self.fc7 = nn.Linear()
        # self.fc8 = nn.Linear()

        # self.functions = nn.ModuleDict({
        #     'conv1': [self.conv1, F.relu, self.bn1],
        #     'pool1': nn.MaxPool2d(3),
        #     'conv2': [self.conv2, F.relu, self.bn2],
        #     'pool2': nn.MaxPool2d(3),
        #     'conv3': [self.conv3, F.relu],
        #     'conv4': [self.conv4, F.relu],
        #     'conv5': [self.conv5, F.relu],
        #     'pool5': nn.MaxPool2d(3),
        #     'fc6': [self.fc6, F.relu],
        #     'fc7': [self.fc7, F.relu],
        #     'fc8': [self.fc8],
        #     'prob': [nn.Softmax],
        # })

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=96, pad=3, stride=1, ksize=7).to_gpu()
            self.bn1 = L.BatchNormalization(96).to_gpu(),
            self.conv2 = L.Convolution2D(in_channels=None, out_channels=128, pad=2, stride=1, ksize=5).to_gpu()
            self.bn2 = L.BatchNormalization(128).to_gpu(),
            self.conv3 = L.Convolution2D(in_channels=None, out_channels=256, pad=1, stride=1, ksize=3).to_gpu()
            self.conv4 = L.Convolution2D(in_channels=None, out_channels=384, pad=1, stride=1, ksize=3).to_gpu()
            self.conv5 = L.Convolution2D(in_channels=None, out_channels=256, pad=1, stride=1, ksize=3).to_gpu()
            self.fc6 = L.Linear(None, 2048).to_gpu()
            self.fc7 = L.Linear(None, 512).to_gpu()
            self.fc8 = L.Linear(None, 3).to_gpu()
        # End_With

        self.functions = collections.OrderedDict([
            ("conv1", [self.conv1, F.relu, self.bn1]),
            ("pool1", [_MaxPooling2D]),
            ("conv2", [self.conv2, F.relu, self.bn2]),
            ("pool2", [_MaxPooling2D]),
            ("conv3", [self.conv3, F.relu]),
            ("conv4", [self.conv4, F.relu]),
            ("conv5", [self.conv5, F.relu]),
            ("pool5", [_MaxPooling2D]),
            ("fc6", [self.fc6, F.relu]),
            ("fc7", [self.fc7, F.relu]),
            ("fc8", [self.fc8]),
            ("prob", [F.softmax]),
        ])  # End_Functions
    # End_Constructor

    def __call__(self, x, layers=[gv.G.TargetLayer]):
        h = Variable(gv.G.xp.array(x)) if isinstance(x, gv.G.xp.ndarray) else x
        activations = {"input": h}  # 各階層のデータ全部
        #targetLayers = set([gv.G.TargetLayer])
        targetLayers = set(layers)
        for key, funcs in self.functions.items():
            # print("key:{}".format(key))
            #for f in funcs: print("funcs:{}".format(f))
            if len(targetLayers) == 0:
                break
            for func in funcs:
                #print("type(h) : {}".format(type(h.data)))
                if isinstance(func, tuple):
                    h = func[0](h)
                else:
                    h = func(h)
            if key in targetLayers:
                activations[key] = h
                targetLayers.remove(key)
        return activations
    # End_Method

# End_Class


def _MaxPooling2D(x):
    return F.max_pooling_2d(x, ksize=3)
# End_Func
